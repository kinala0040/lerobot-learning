# 视频解码深度探索

> 学习日期：2026-09-22  
> 目标：理解 LeRobot 如何从 MP4 视频中提取图像帧

---

## 1. 为什么用视频而不是单独的图像？

### 方案对比

| 存储方式 | 文件大小 | 文件数量 | 随机访问速度 | 适用场景 |
|---------|---------|---------|-------------|---------|
| **PNG 序列** | ~5 GB | 25,650 个 | 快 | 小数据集 |
| **MP4 视频** | ~6.5 MB | 1 个 | 中等 | 大数据集 ✅ |
| **Parquet (压缩图像)** | ~100 MB | 1 个 | 慢 | 不常用 |

### LeRobot 的选择：MP4

**优势**：
- **压缩率高**：5 GB → 6.5 MB（压缩 800 倍！）
- **单文件管理**：方便上传/下载
- **标准格式**：所有平台都支持

**劣势**：
- 需要解码（CPU 开销）
- 随机访问比 PNG 慢

---

## 2. 视频编码原理

### 视频压缩如何工作？

```
原理：利用帧间相似性

帧 0 (关键帧):   ████████████  (完整图像，100 KB)
                  ↓ 相似度 95%
帧 1 (差分帧):    ░░░▓░░░░░░░░  (只存储差异，5 KB)
                  ↓ 相似度 95%
帧 2 (差分帧):    ░░░░▓░░░░░░░  (只存储差异，5 KB)
                  ↓
...

总大小: 100 + 5×N KB (远小于 100×N KB)
```

**关键概念**：
- **关键帧 (I-frame)**：完整的图像（每 10-30 帧一个）
- **差分帧 (P-frame)**：只存储与前一帧的差异
- **双向差分帧 (B-frame)**：参考前后帧计算差异

---

## 3. 解码流程

### 从 MP4 提取一帧的步骤

```python
# 伪代码
def get_frame(video_path, frame_index):
    # 1. 打开视频文件
    video = open_video(video_path)
    
    # 2. 定位到目标帧
    video.seek(frame_index)
    
    # 3. 解码
    if frame_index 是关键帧:
        # 直接解码
        frame = decode_keyframe(video)
    else:
        # 需要先解码前面的关键帧，再应用差分
        keyframe = find_previous_keyframe(frame_index)
        frame = decode_from_keyframe(keyframe, frame_index)
    
    # 4. 返回图像 (NumPy 数组)
    return frame  # shape: [H, W, 3]
```

### 为什么随机访问比 PNG 慢？

```
PNG 序列：
  读取帧 1000 → 直接打开 frame_1000.png → 完成 (1 次 I/O)

MP4 视频：
  读取帧 1000 → 找到前一个关键帧（如帧 990）
              → 解码关键帧
              → 应用差分 990→991→...→1000 → 完成 (1 次 I/O + 10 次解码)
```

---

## 4. LeRobot 的视频解码实现

### 使用的库

LeRobot 支持两种视频解码器：

1. **torchcodec**（默认，速度快）
   - PyTorch 官方的视频解码库
   - GPU 加速解码
   - 但需要 FFmpeg

2. **pyav**（备用，兼容性好）
   - Python 绑定的 FFmpeg
   - 纯 CPU 解码
   - 更稳定

### 缓存机制

```python
class VideoDecoder:
    def __init__(self, video_path):
        self.video = open_video(video_path)
        self.cache = LRU_Cache(maxsize=128)  # 缓存最近访问的 128 帧
    
    def get_frame(self, frame_index):
        # 1. 检查缓存
        if frame_index in self.cache:
            return self.cache[frame_index]  # 命中！直接返回
        
        # 2. 解码
        frame = self.video.decode(frame_index)
        
        # 3. 缓存
        self.cache[frame_index] = frame
        
        return frame
```

**为什么需要缓存？**
- 训练时常常访问连续的帧（如帧 100-110）
- 缓存避免重复解码
- LRU（Least Recently Used）策略：自动淘汰最久未用的帧

---

## 5. 训练时的访问模式

### Episode 采样

```python
# 训练时的典型流程
for epoch in range(100):
    for batch in dataloader:
        # 1. 随机采样一个 Episode
        episode_idx = random.choice(range(206))
        
        # 2. 在 Episode 内随机采样一个起始帧
        start_frame = random.randint(0, episode_length - 10)
        
        # 3. 提取连续的 10 帧
        frames = []
        for i in range(start_frame, start_frame + 10):
            frame = video_decoder.get_frame(i)
            frames.append(frame)
        
        # 4. 训练
        model(frames)
```

**访问模式特点**：
- ✅ 连续访问（start_frame, start_frame+1, ..., start_frame+9）
- ✅ 缓存友好（相邻帧很可能在缓存中）
- ❌ 跨 Episode 访问不连续（Episode 5 → Episode 50）

---

## 6. 性能优化技巧

### 技巧 1：预加载（Prefetching）

```python
# 主线程
for batch in dataloader:
    # 训练当前 batch
    model(batch)

# 后台线程（同时进行）
next_batch = load_next_batch()  # 解码下一个 batch 的视频帧
```

### 技巧 2：多进程解码

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 4 个进程并行解码视频
    prefetch_factor=2  # 每个进程预加载 2 个 batch
)
```

### 技巧 3：降采样

```python
# 原始图像：480×640
# 训练用图像：96×96

frame = video_decoder.get_frame(i)  # [480, 640, 3]
frame = resize(frame, (96, 96))     # [96, 96, 3] ← 减少 27 倍数据量
```

---

## 7. 实际性能测试

### 单帧解码速度

| 解码器 | 首次访问 | 缓存命中 | GPU 加速 |
|--------|---------|---------|---------|
| torchcodec | ~5 ms | ~0.1 ms | ✅ |
| pyav | ~10 ms | ~0.1 ms | ❌ |
| PNG | ~2 ms | ~0.1 ms | - |

### 批量解码（32 帧）

| 解码器 | 单线程 | 4 线程 | 性能 |
|--------|--------|--------|------|
| torchcodec | 160 ms | 50 ms | 最快 |
| pyav | 320 ms | 100 ms | 中等 |
| PNG | 64 ms | 20 ms | 最快（但文件大）|

---

## 8. 数据流水线

### 完整的数据加载流程

```
1. 采样 Episode
   ↓
2. 读取 Parquet（状态、动作）← 0.0255 秒（很快！）
   ↓
3. 解码视频帧 ← 0.05-0.1 秒（瓶颈）
   ↓
4. 数据增强（旋转、裁剪）
   ↓
5. 归一化（[0,255] → [0,1]）
   ↓
6. 转换为 Tensor
   ↓
7. 组成 Batch
   ↓
8. 送入模型
```

**瓶颈**：视频解码（步骤 3）

**优化策略**：
- 多进程并行解码
- 缓存最近访问的帧
- 预加载下一个 batch

---

## 9. 核心理解：为什么随机访问慢？

### 视频文件的真实结构

**重要**：`file-000.mp4` 不是一个 Episode，而是 **206 个 Episode 拼接在一起**！

```
file-000.mp4 (25650 帧)
┌─────────────────────────────────────────────────────────────┐
│ Ep0   │ Ep1  │ Ep2   │ ... │ Ep50  │ ... │ Ep120 │ ... │Ep205│
│ 0-160 │161-278│279-419│ ... │6000-  │ ... │15234- │ ... │25649│
└─────────────────────────────────────────────────────────────┘

类比：
  file-000.mp4 = 一部电影
  每个 Episode = 一个场景
  206 个场景拼接在一起
```

---

### 两种"随机"的本质区别

#### ❌ 帧级别随机（错误做法）

```python
# 从所有 25650 帧中随机采样
frame_indices = random.sample(range(25650), 32)
# 例如：[100, 15234, 420, 8900, ...]

for idx in frame_indices:
    frame = dataset[idx]

访问模式：
  帧 100 (Ep0) → 帧 15234 (Ep120) → 帧 420 (Ep3) → 帧 8900 (Ep70)
  
问题：
  - 每次跳跃都要重新定位视频位置
  - 需要找到新的关键帧
  - 缓存失效
  
结果：平均 3745 ms/帧（极慢！）
```

**类比**：就像看电影时不停地拖动进度条
```
[░░░░░▓░░░░░░░░░░░] → 快进到这里
[░░░░░░░░░░░░▓░░░░] → 又快进到这里  
[░▓░░░░░░░░░░░░░░░] → 又快退回这里
播放器卡顿！
```

---

#### ✅ Episode 级别随机（正确做法）

```python
# 随机打乱 Episode 顺序
episodes = list(range(206))
random.shuffle(episodes)
# 例如：[120, 3, 50, 88, 0, 5, ...]

for ep_idx in episodes:
    start = episode_from_index[ep_idx]
    end = episode_to_index[ep_idx]
    
    # Episode 内顺序读取
    for frame_idx in range(start, end):
        frame = dataset[frame_idx]

访问模式：
  Episode 120: 帧 15000 → 15001 → 15002 → ... → 15167 (顺序！)
  Episode 3:   帧 420 → 421 → 422 → ... → 578 (顺序！)
  Episode 50:  帧 6000 → 6001 → 6002 → ... → 6159 (顺序！)
  
优点：
  - Episode 内部顺序访问 → 利用视频缓存
  - 视频解码器不需要重新定位
  - 连续帧已在缓存中
  
结果：平均 0.9 ms/帧（快 4150 倍！）
```

**类比**：按章节看电影
```
章节顺序是随机的：
  先看第20章（从头到尾）✅ 流畅
  再看第3章（从头到尾）✅ 流畅
  再看第50章（从头到尾）✅ 流畅
  
每章都是顺序播放，不卡顿！
```

---

### 为什么 Episode 级别随机就够了？

#### 训练需要的随机性

```python
模型需要学习：
  "给定状态 S，应该采取动作 A"

不需要学习：
  "Episode 0 后面总是 Episode 1"  ← 这是错误的模式

只要 Episode 顺序是随机的，模型就学不到这种假的模式！
```

#### 随机性对比

| 维度 | 帧级别随机 ❌ | Episode 级别随机 ✅ |
|------|--------------|-------------------|
| **随机粒度** | 每一帧独立随机 | Episode 顺序随机 |
| **Episode 内部** | 跳来跳去 | **顺序读取** |
| **视频解码** | 每次都要重新定位 | 利用缓存 |
| **速度** | 3745 ms/帧（超慢）| 0.9 ms/帧（快）|
| **随机性** | 过强（浪费） | 足够（高效）|
| **排列数** | 25650! (天文数字) | 206! (仍是天文数字) |

---

### 实际性能测试结果

从我们的实验：

```
顺序访问 100 帧:
  总耗时: 0.09 秒
  平均: 0.9 ms/帧
  
随机访问 10 帧:
  总耗时: 37.45 秒
  平均: 3745 ms/帧
  
性能差异: 4150 倍！
```

**为什么差距这么大？**

```
顺序访问（帧 0→1→2→...）:
  帧 0 (关键帧): 完整解码 [10 ms]
  帧 1 (差分帧): 基于帧 0 计算 [1 ms] ✅ 快！
  帧 2 (差分帧): 基于帧 1 计算 [1 ms] ✅ 快！
  平均: ~1 ms/帧

随机访问（帧 100→15234→420→...）:
  帧 100:   从头解码到帧 100 [0.06 s]
  帧 15234: 从头解码到帧 15234 [4.3 s] ← 跳很远！
  帧 420:   从头解码到帧 420 [0.4 s] ← 又跳回去！
  平均: ~1.6 s/帧
```

---

## 10. 训练的最佳实践

### ✅ 推荐的训练流程

```python
for epoch in range(100):
    # 步骤 1: 随机打乱 Episode 顺序
    episodes = list(range(206))
    random.shuffle(episodes)
    
    # 步骤 2: 按 Episode 顺序处理
    for ep_idx in episodes:
        # 获取 Episode 的帧范围
        start = dataset.episode_data_index['from'][ep_idx]
        end = dataset.episode_data_index['to'][ep_idx]
        
        # 步骤 3: Episode 内顺序读取
        for frame_idx in range(start, end):
            frame = dataset[frame_idx]
            model.train(frame)

结果：
  - Episode 之间：随机（满足训练需求）✅
  - Episode 内部：顺序（利用视频缓存）✅
  - 速度：快 4150 倍！✅
```

### ❌ 要避免的做法

```python
# 永远不要这样做！
for batch in range(num_batches):
    # 全局随机采样
    indices = random.sample(range(25650), 32)
    
    for idx in indices:
        frame = dataset[idx]  # ← 随机跳跃，慢 4150 倍！
```

---

## 11. 核心总结

### 视频解码的关键理解

```
1. 视频文件 = 多个 Episode 拼接
   file-000.mp4 ≠ 一个 Episode
   file-000.mp4 = Episode 0 + Episode 1 + ... + Episode 205

2. 视频压缩原理 = 关键帧 + 差分帧
   顺序访问 → 利用差分帧 → 快
   随机访问 → 重新解码关键帧 → 慢

3. 训练的随机性 = Episode 级别就够
   Episode 顺序随机 → 满足训练需求
   Episode 内部顺序 → 利用视频特性
   
4. 性能对比：
   顺序访问: 0.9 ms/帧
   随机访问: 3745 ms/帧
   差异: 4150 倍！
```

### 类比总结

```
帧级别随机 = 看书时每次随机翻一页
  → 翻来翻去，找不到页码，很慢 ❌

Episode 级别随机 = 随机选章节，但每章从头读到尾
  → 章节顺序乱，但每章顺序读，快很多 ✅
```

---

## 12. 下一步

我们将创建脚本来：
1. 提取视频的单帧并保存为图像
2. 测试不同解码器的性能
3. 可视化缓存命中率
4. 实现简单的 Episode 采样器

---

## 总结

### LeRobot 的视频解码策略

✅ **存储**：MP4 格式（压缩率高）  
✅ **解码**：torchcodec/pyav（按需解码）  
✅ **缓存**：LRU 缓存（避免重复解码）  
✅ **并行**：多进程解码（提高吞吐量）  

### 关键权衡

```
PNG 序列 vs MP4:
  - 存储空间：MP4 胜（800 倍压缩）
  - 访问速度：PNG 胜（直接读取）
  - 最终选择：MP4（空间优先）
```
