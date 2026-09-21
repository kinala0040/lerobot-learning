# LeRobot 数据存储深度解析

## 学习目标
理解 LeRobot 如何高效存储和加载机器人数据。

---

## 1. 为什么需要特殊的存储格式？

### 问题：机器人数据的特点

```
挑战 1：数据量大
  - 206 个 Episode × 平均 125 帧 = 25,650 帧
  - 每帧：图像 (96×96×3) + 状态 (2) + 动作 (2)
  - 图像占用空间：96 × 96 × 3 × 4 bytes = 110 KB/帧
  - 总大小：25,650 帧 × 110 KB ≈ 2.8 GB

挑战 2：访问模式复杂
  - 训练时：随机采样 Episode
  - 每次采样：需要连续的 N 帧（时间窗口）
  - 不能每次都加载整个数据集到内存

挑战 3：多模态数据
  - 图像（视频文件）
  - 状态（数值）
  - 动作（数值）
  - 元数据（Episode 边界、任务描述）

挑战 4：分布式存储
  - 数据集托管在 Hugging Face Hub
  - 需要支持增量下载
  - 需要支持本地缓存
```

### 解决方案：LeRobot 的存储架构

```
LeRobot 使用：
  1. Parquet 文件（列式存储） → 存储数值数据（状态、动作、时间戳）
  2. MP4 视频文件 → 存储图像序列
  3. JSON 元数据 → 存储 Episode 边界和配置
  4. 分块机制 → 支持增量加载
```

---

## 2. Parquet 文件格式

### 什么是 Parquet？

```
Parquet = Apache 开发的列式存储格式

行式存储（CSV）：
  Row 0: [episode_0, frame_0, state_x_0, state_y_0, action_x_0, action_y_0]
  Row 1: [episode_0, frame_1, state_x_1, state_y_1, action_x_1, action_y_1]
  Row 2: [episode_0, frame_2, state_x_2, state_y_2, action_x_2, action_y_2]
  
  读取：一次读一整行（所有列）
  问题：如果只需要 action 列，也要读取所有列

列式存储（Parquet）：
  Column episode_index: [0, 0, 0, 1, 1, 1, ...]
  Column frame_index:   [0, 1, 2, 0, 1, 2, ...]
  Column state_x:       [250, 255, 260, ...]
  Column state_y:       [300, 305, 310, ...]
  Column action_x:      [255, 260, 265, ...]
  Column action_y:      [305, 310, 315, ...]
  
  读取：只读需要的列
  优势：压缩率高、查询快
```

### Parquet 的优势

| 特性 | CSV | Parquet |
|------|-----|---------|
| 文件大小 | 100 MB | 30 MB（压缩） |
| 读取全部数据 | 1.0x | 0.8x（快 20%） |
| 读取单列 | 1.0x | **0.1x**（快 10 倍） |
| 支持数据类型 | 字符串 | int, float, bool, datetime |
| 支持嵌套数据 | ❌ | ✅ |

---

## 3. LeRobot 的数据文件结构

### 目录结构

```
~/.cache/huggingface/hub/datasets--lerobot--pusht/
├── snapshots/
│   └── <commit_hash>/
│       ├── meta/
│       │   └── episodes.parquet        # Episode 元数据
│       ├── data/
│       │   └── chunk-000/
│       │       └── data.parquet        # 状态和动作数据
│       ├── videos/
│       │   └── observation.image/
│       │       └── chunk-000/
│       │           └── video.mp4       # 图像序列
│       ├── meta.json                   # 数据集配置
│       └── stats.json                  # 统计信息（归一化用）
```

### 文件作用

| 文件 | 内容 | 大小 | 作用 |
|------|------|------|------|
| `meta/episodes.parquet` | Episode 边界、任务描述 | ~100 KB | 快速定位 Episode |
| `data/chunk-000/data.parquet` | 状态、动作、时间戳 | ~10 MB | 数值数据 |
| `videos/.../video.mp4` | 图像序列（压缩） | ~100 MB | 视觉观测 |
| `meta.json` | 数据集配置 | ~5 KB | 加载参数 |
| `stats.json` | 均值、标准差 | ~2 KB | 归一化 |

---

## 4. 分块（Chunking）机制

### 为什么需要分块？

```
问题：
  - pusht 只有 206 个 Episode（25,650 帧）
  - 但大型数据集可能有 10,000+ 个 Episode（100 万帧）
  - 不能把所有数据放在一个文件里

解决方案：分块存储
  - 每 N 个 Episode 存一个 chunk
  - 训练时只加载需要的 chunk
  - 支持增量下载（只下载需要的 chunk）
```

### 分块示例

```
小数据集（pusht）：
  data/chunk-000/data.parquet          # Episode 0-205（全部）
  videos/observation.image/chunk-000/  # Episode 0-205

大数据集（10,000 Episodes）：
  data/chunk-000/data.parquet          # Episode 0-999
  data/chunk-001/data.parquet          # Episode 1000-1999
  data/chunk-002/data.parquet          # Episode 2000-2999
  ...
  videos/observation.image/chunk-000/  # Episode 0-999
  videos/observation.image/chunk-001/  # Episode 1000-1999
  ...
```

---

## 5. Episode 索引机制

### meta/episodes.parquet 的结构

```python
# 加载 Episode 元数据
import pandas as pd
episodes = pd.read_parquet("meta/episodes.parquet")

print(episodes.head())
```

输出：
```
   episode_index  dataset_from_index  dataset_to_index  length  tasks
0              0                   0               161     161  [Push...]
1              1                 161               279     118  [Push...]
2              2                 279               420     141  [Push...]
```

### 快速定位 Episode

```python
# 问题：我想访问 Episode 5
ep_idx = 5

# 方法 1：遍历所有帧（慢）
for i in range(len(dataset)):
    if dataset[i]['episode_index'] == ep_idx:
        # 找到了
        pass
# 问题：需要遍历 25,650 帧！

# 方法 2：使用 Episode 索引（快）
from_idx = episodes.loc[ep_idx, 'dataset_from_index']  # O(1)
to_idx = episodes.loc[ep_idx, 'dataset_to_index']      # O(1)

# 直接访问
for i in range(from_idx, to_idx):
    frame = dataset[i]
# 只访问 Episode 5 的帧！
```

---

## 6. 数据加载流程

### 完整的加载流程

```python
from lerobot.datasets import LeRobotDataset

# 1. 加载数据集
dataset = LeRobotDataset('lerobot/pusht')

# 背后发生了什么：
# Step 1: 下载 meta.json（如果本地没有）
# Step 2: 解析 meta.json，获取文件列表
# Step 3: 下载 meta/episodes.parquet
# Step 4: 下载 data/chunk-000/data.parquet
# Step 5: 下载 videos/observation.image/chunk-000/video.mp4
# Step 6: 加载统计信息（stats.json）
# Step 7: 构建 Episode 索引

# 2. 访问一帧
frame = dataset[100]

# 背后发生了什么：
# Step 1: 查找 Episode 索引（100 属于哪个 Episode？）
# Step 2: 从 Parquet 读取第 100 行（状态、动作）
# Step 3: 从视频解码第 100 帧（图像）
# Step 4: 组合成一个字典返回
```

---

## 7. 视频解码机制

### 为什么用视频而不是图像序列？

```
方案 A：存储为单独的 PNG 文件
  - 优点：简单，易于查看
  - 缺点：文件数量多（25,650 个 PNG），占用空间大（~5 GB）

方案 B：存储为 MP4 视频
  - 优点：压缩率高（~100 MB），文件数量少（1 个文件）
  - 缺点：需要解码，随机访问稍慢

LeRobot 选择：MP4（空间优先）
```

### 视频解码流程

```python
# 伪代码
class VideoDecoder:
    def __init__(self, video_path):
        self.video = open_video(video_path)  # 打开视频文件
        self.cache = {}  # 缓存最近访问的帧
    
    def get_frame(self, frame_idx):
        # 1. 检查缓存
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        
        # 2. 解码帧
        self.video.seek(frame_idx)  # 定位到第 N 帧
        frame = self.video.read()   # 解码
        
        # 3. 缓存
        self.cache[frame_idx] = frame
        
        return frame
```

---

## 8. 下一步：实践探索

### 我们将创建的脚本

1. **探索 Parquet 文件结构**
   - 查看文件大小
   - 读取列信息
   - 对比 Parquet vs CSV

2. **理解分块机制**
   - 查看 chunk 边界
   - 模拟增量加载

3. **视频解码实验**
   - 提取单帧
   - 测试随机访问速度

4. **构建自定义 DataLoader**
   - Episode 采样
   - 批量加载
   - 性能优化

---

## 总结

LeRobot 的数据存储架构：
- **Parquet**：高效存储数值数据（列式、压缩）
- **MP4**：高效存储图像（视频压缩）
- **分块**：支持大规模数据集（增量下载）
- **索引**：快速定位 Episode（O(1) 查找）

这些设计让 LeRobot 能够：
- 高效存储大规模数据集
- 支持快速训练（随机采样 + 批量加载）
- 云端分发友好（Hugging Face Hub）
