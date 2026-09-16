# LeRobot 数据基础知识

> 学习日期：2026-09-10  
> 数据集：lerobot/pusht  
> 状态：✅ 已完成数据加载和基础概念理解

---

## 一、核心概念

### 1.1 数据层级结构

```
Dataset (数据集)
├── Episode 0 (第 1 次任务执行)
│   ├── Frame 0 (第 0 时刻)
│   ├── Frame 1 (第 1 时刻)
│   └── ...
├── Episode 1 (第 2 次任务执行)
│   ├── Frame 161
│   ├── Frame 162
│   └── ...
└── ...
```

**类比理解**：
- **Dataset** = 一整季电视剧
- **Episode** = 一集
- **Frame** = 一集中的每一帧画面

---

### 1.2 关键术语定义

| 术语 | 英文 | 定义 | 示例 |
|------|------|------|------|
| **数据集** | Dataset | 所有机器人数据的集合 | lerobot/pusht 有 206 个 Episode |
| **Episode** | Episode | 一次完整的任务执行，从开始到结束 | "把方块推到目标位置"的一次尝试 |
| **帧** | Frame | Episode 中某一时刻的快照 | Episode 0 的第 50 帧 |
| **观测** | Observation | 机器人"感知到"的信息 | 相机图像 + 关节角度 |
| **状态** | State | 机器人的物理状态 | 位置 (x, y)、速度、关节角度 |
| **动作** | Action | 机器人执行的控制指令 | 移动方向 (Δx, Δy)、目标关节角度 |
| **时间戳** | Timestamp | 该帧的时间点 | 0.0, 0.1, 0.2, ... (秒) |
| **FPS** | Frames Per Second | 每秒记录的帧数 | 10 FPS = 每 0.1 秒记录一帧 |

---

## 二、pusht 数据集分析

### 2.1 数据集统计信息

```python
数据集名称: lerobot/pusht
总帧数: 25,650
Episode 数量: 206
FPS (每秒帧数): 10
平均每个 Episode: 124 帧 ≈ 12.4 秒
```

**计算推导**：
- 平均 Episode 长度 = 25,650 ÷ 206 ≈ 124 帧
- 平均任务时长 = 124 帧 ÷ 10 FPS = 12.4 秒

---

### 2.2 Episode 和 Frame 的索引关系

```python
Episode 0 的帧范围: [0, 161)  # 第 0-160 帧
Episode 0 长度: 161 帧 = 16.1 秒
```

**理解**：
- Episode 0 占用全局索引 0-160（共 161 帧）
- Episode 1 会从全局索引 161 开始
- `[0, 161)` 是左闭右开区间

**访问方式**：
```python
dataset[0]      # 全局第 0 帧（Episode 0 的第 0 帧）
dataset[160]    # Episode 0 的最后一帧
dataset[161]    # Episode 1 的第一帧
```

---

### 2.3 数据字段（Features）

每一帧包含以下字段：

| 字段名 | 数据类型 | Shape | 含义 |
|--------|---------|-------|------|
| `observation.image` | torch.Tensor | `[3, 96, 96]` | RGB 图像，3 通道，96×96 像素 |
| `observation.state` | torch.Tensor | `[2]` | 机器人末端位置 (x, y) |
| `action` | torch.Tensor | `[2]` | 机器人动作 (Δx, Δy) |
| `episode_index` | torch.Tensor | `[]` (标量) | 该帧属于哪个 Episode |
| `frame_index` | torch.Tensor | `[]` (标量) | 该帧在 Episode 中的索引 |
| `timestamp` | torch.Tensor | `[]` (标量) | 时间戳（秒） |
| `next.reward` | torch.Tensor | `[]` (标量) | 强化学习奖励值 |
| `next.done` | torch.Tensor | `[]` (布尔) | Episode 是否结束 |
| `next.success` | torch.Tensor | `[]` (布尔) | 任务是否成功 |
| `task` | str | - | 任务描述文本 |
| `task_index` | torch.Tensor | `[]` (标量) | 任务类型索引 |
| `index` | torch.Tensor | `[]` (标量) | 全局帧索引 |

---

### 2.4 第 0 帧实例分析

```python
frame_0 = dataset[0]

# 图像：机器人"看到"的画面
frame_0['observation.image']  # shape=[3, 96, 96], RGB 图像

# 状态：机器人末端的 (x, y) 坐标
frame_0['observation.state']  # shape=[2], 例如 [0.5, 0.3]

# 动作：机器人的移动指令
frame_0['action']  # shape=[2], 例如 [0.01, -0.02]

# 任务描述
frame_0['task']  # "Push the T-shaped block onto the T-shaped target."
```

**pusht 任务理解**：
- **Observation.image**：俯视图，显示 T 形方块、目标位置、机器人末端
- **Observation.state**：机器人末端在 2D 平面的 (x, y) 位置
- **Action**：机器人末端下一步的移动方向和距离 (Δx, Δy)
- **任务目标**：通过移动末端，把 T 形方块推到 T 形目标区域

---

## 三、重要技术细节

### 3.1 PyTorch 数据格式

LeRobot 数据集返回的所有数值都是 **PyTorch Tensor**：

```python
type(frame_0['observation.image'])  # <class 'torch.Tensor'>
```

**图像格式约定**：
- PyTorch 格式：`[C, H, W]`（通道优先）
- 通用格式：`[H, W, C]`（高度、宽度、通道）
- `observation.image` 的 shape `[3, 96, 96]` = `[C=3, H=96, W=96]`

### 3.2 Observation vs State vs Action

```
┌─────────────────────────────────────────┐
│          Observation (观测)              │
│  ┌─────────────────┐  ┌──────────────┐ │
│  │ observation.image│  │observation.  │ │
│  │  (相机图像)      │  │state (状态)  │ │
│  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────┘
              ↓
         [ 策略模型 ]
              ↓
        ┌──────────┐
        │  Action  │  ← 机器人执行的控制指令
        └──────────┘
```

- **Observation**（观测）= Image（图像）+ State（状态）
  - 机器人"感知"到的所有信息
  - 是策略模型的**输入**
  
- **State**（状态）
  - 机器人自身的物理状态（位置、速度、关节角度等）
  - 是 Observation 的一部分
  
- **Action**（动作）
  - 机器人执行的控制指令
  - 是策略模型的**输出**

---

## 四、数据加载代码

### 4.1 加载完整数据集

```python
from lerobot.datasets import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset('lerobot/pusht')

# 查看基本信息
print(f'总帧数: {len(dataset)}')
print(f'Episode 数量: {dataset.num_episodes}')
print(f'FPS: {dataset.fps}')
```

### 4.2 访问 Episode 信息

```python
# 获取 Episode 0 的帧范围
from_idx = dataset.meta.episodes['dataset_from_index'][0]
to_idx = dataset.meta.episodes['dataset_to_index'][0]

print(f'Episode 0: 帧 [{from_idx}, {to_idx})')
print(f'Episode 0 长度: {to_idx - from_idx} 帧')
```

### 4.3 访问单帧数据

```python
# 获取第 0 帧
frame = dataset[0]

# 查看所有字段
for key, value in frame.items():
    if hasattr(value, 'shape'):
        print(f'{key}: shape={value.shape}')
    else:
        print(f'{key}: {value}')
```

---

## 五、环境配置记录

### 5.1 安装依赖

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 安装 LeRobot 基础依赖
cd ~/lerebot/lerobot
uv sync --locked

# 安装数据集相关依赖
uv sync --locked --extra dataset
```

### 5.2 代理配置（如需要）

```bash
# 设置代理
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
```

### 5.3 已知问题

- ⚠️ **FFmpeg 警告**：`Could not load libtorchcodec`
  - **影响**：无，系统自动回退到 `pyav` 解码器
  - **原因**：未安装 FFmpeg 或版本不匹配
  - **处理**：可暂时忽略，不影响学习

---

## 六、学习检查清单

- [x] 理解 Dataset、Episode、Frame 的层级关系
- [x] 理解 Observation、State、Action 的区别
- [x] 成功加载 lerobot/pusht 数据集
- [x] 查看数据集统计信息（总帧数、Episode 数量、FPS）
- [x] 访问单帧数据并理解每个字段的含义
- [x] 理解 PyTorch Tensor 格式和图像的 shape
- [ ] 可视化图像数据（下一步）
- [ ] 理解时间戳和帧的对应关系
- [ ] 实现数据质量检查

---

## 七、面试准备

### 可以回答的问题：

**Q1: 什么是 Episode？**
> Episode 是机器人执行一次完整任务的过程，从任务开始到结束（成功或失败）。例如"把方块推到目标位置"的一次尝试就是一个 Episode。

**Q2: 一帧数据包含什么？**
> 一帧包含该时刻的观测（图像 + 机器人状态）、执行的动作、时间戳、奖励信号、以及 Episode 和任务的元信息。

**Q3: Observation 和 State 有什么区别？**
> Observation 是机器人感知到的所有信息，包括图像和状态；State 是机器人自身的物理状态（位置、速度等），是 Observation 的一部分。

**Q4: 为什么图像是 [3, 96, 96] 而不是 [96, 96, 3]？**
> 这是 PyTorch 的 channel-first 约定，shape 是 [C, H, W]。训练时输入神经网络的图像批次是 [B, C, H, W]，这种格式在 GPU 上计算效率更高。

---

## 八、简历可写内容（当前阶段）

**暂时不要写**，因为只是完成了数据加载。等完成数据质量检查、数据集切分、Benchmark 后再写。

---

## 九、下一步计划

1. **可视化数据**：将图像 Tensor 转换为图片并查看
2. **时间序列分析**：理解时间戳、帧率、Episode 时长的关系
3. **数据质量检查**：检查缺失值、异常值、时间戳连续性
4. **Episode 统计**：分析 Episode 长度分布、成功率

---

**笔记更新日志**：
- 2026-09-10：创建笔记，完成数据加载和基础概念学习
