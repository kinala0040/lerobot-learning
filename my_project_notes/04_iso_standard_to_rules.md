# 基于国际标准的数据质量规则定义

> 学习日期：2026-09-10  
> 参考标准：ISO/IEC 5259 (AI/ML 数据质量标准)  
> 应用场景：LeRobot 机器人轨迹数据质量检查

---

## 一、为什么需要标准化的规则定义？

### 1.1 问题

之前的困惑：
- ❓ **阈值是"拍脑袋"定的吗？**（Episode < 30 帧、动作变化 > 100 像素）
- ❓ **不同人定义的规则差异巨大，如何统一？**
- ❓ **面试时如何解释规则的合理性？**

### 1.2 解决方案

**使用国际标准作为框架**：
```
ISO/IEC 5259 标准（抽象原则）
         ↓
   映射到数据结构
         ↓
   定义具体检查规则
         ↓
   设置数值阈值
```

这样做的好处：
- ✅ **有理有据**：基于国际标准，不是随意定义
- ✅ **可复用**：框架适用于任何机器人数据集
- ✅ **可沟通**：团队成员、面试官都能理解
- ✅ **可扩展**：新任务只需调整阈值，不需重新设计

---

## 二、ISO/IEC 5259 标准解读

### 2.1 标准概述

**ISO/IEC 5259** 系列：AI 和机器学习数据质量标准

- **ISO/IEC 5259-1**: 数据质量框架和通用原则
- **ISO/IEC 5259-2**: 数据质量管理指南
- **ISO/IEC 5259-3**: 数据质量评估方法

**核心理念**：
> AI/ML 数据质量不只是"没有错误"，而是**适合特定用途（Fitness for Purpose）**的程度。

---

### 2.2 四个核心维度

ISO/IEC 5259 定义的数据质量维度：

| 维度 | 英文 | 定义 | 问题示例 |
|------|------|------|---------|
| **完整性** | Completeness | 数据是否包含所有必需的信息 | 缺失字段、Episode 过短 |
| **一致性** | Consistency | 数据内部是否逻辑一致、无矛盾 | 时间戳跳跃、动作突变 |
| **适用性** | Fitness for Purpose | 数据是否适合下游任务使用 | 图像模糊、任务标签错误 |
| **可度量性** | Measurability | 数据的数值是否可靠、可复现 | 超出范围、量纲错误 |

**其他相关维度**（ISO/IEC 5259 还包括，但我们暂不使用）：
- Accuracy（准确性）：数据与真实世界的偏差
- Timeliness（时效性）：数据是否过时
- Accessibility（可访问性）：数据获取的容易程度

---

## 三、从标准到 LeRobot 数据结构的映射

### 3.1 LeRobot 数据结构

根据 [LeRobot Dataset v3 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx)：

```python
LeRobotDataset 结构：
│
├── Episode 级别
│   ├── episode_index: int
│   ├── task: str
│   ├── length: int (from_index → to_index)
│   └── success: bool
│
└── Frame 级别（每个 Episode 的每一帧）
    ├── observation.image: [C, H, W]
    ├── observation.state: [state_dim]
    ├── action: [action_dim]
    ├── timestamp: float
    ├── frame_index: int
    └── next.reward / next.done / next.success
```

**关键特性**：
- **多模态**：图像 + 状态 + 动作
- **时序**：有时间戳和顺序
- **分组**：按 Episode 组织
- **标注**：有任务标签和成功标志

---

### 3.2 标准维度 → 数据结构 → 检查规则

#### 维度 1：完整性 (Completeness)

**ISO/IEC 5259 定义**：
> 数据集是否包含完成任务所需的所有信息

**映射到 LeRobot**：
```
完整性问题：
├── Episode 级别
│   ├── Episode 数量太少（< 100）
│   ├── Episode 长度过短（< 30 帧）
│   └── 缺少必要的 Episode（某些任务类型缺失）
│
└── Frame 级别
    ├── 缺失字段（observation.image 为空）
    ├── 缺失帧（时间戳有跳跃）
    └── 数据为空（NaN、None）
```

**具体规则**：

| 规则 ID | 检查项 | 阈值 | 代码示例 |
|---------|--------|------|---------|
| `missing_fields` | 必需字段是否存在 | 0 个缺失 | `'observation.image' not in frame` |
| `nan_values` | 是否有 NaN | 0 个 NaN | `torch.isnan(frame['action']).any()` |
| `too_short` | Episode 最短长度 | ≥ 30 帧 | `episode_length < 30` |

**为什么这些阈值？**
- **30 帧**：
  - pusht 任务理论最短时间：2-3 秒
  - 转换：3 秒 × 10 FPS = 30 帧
  - 少于 30 帧 → 无法学到完整的"开始→执行→结束"轨迹

---

#### 维度 2：一致性 (Consistency)

**ISO/IEC 5259 定义**：
> 数据内部是否逻辑一致、无矛盾、无冲突

**映射到 LeRobot**：
```
一致性问题：
├── 时序一致性
│   ├── 时间戳不递增（t[i] ≥ t[i+1]）
│   ├── 时间戳跳跃过大（dt > 2 × 预期）
│   └── frame_index 不连续
│
├── 动作一致性
│   ├── 相邻帧动作突变（物理上不可能）
│   ├── 动作与状态不匹配（action 指向状态范围外）
│   └── 速度不合理（加速度过大）
│
└── 模态一致性
    ├── 图像与状态不匹配（图像显示位置 A，状态说位置 B）
    └── 动作与观测时序错位
```

**具体规则**：

| 规则 ID | 检查项 | 阈值 | 代码示例 |
|---------|--------|------|---------|
| `timestamp_gap` | 时间戳间隔 | `\|dt - 0.1\| < 0.01` | `abs((t[i] - t[i-1]) - 0.1) > 0.01` |
| `action_jump` | 动作突变 | Δaction < 100 px | `np.linalg.norm(action[i] - action[i-1]) > 100` |
| `velocity_anomaly` | 速度异常 | v < 500 px/s | `np.linalg.norm(action[i] - state[i]) / dt > 500` |

**为什么 100 像素？**
- **人类操作速度**：
  - 工作空间：500×500 像素
  - 合理速度：50 像素/帧（0.1 秒内移动 50 像素 = 500 px/s）
  - 极限速度：100 像素/帧（快速移动）
  - **超过 100 → 不可能是连续动作，是数据错误**

- **数据驱动验证**：
  ```python
  # 统计所有动作变化
  action_diffs = []
  # ... 计算
  print(f"Mean: {np.mean(action_diffs):.1f}")  # ~20-30
  print(f"P95: {np.percentile(action_diffs, 95):.1f}")  # ~70
  print(f"P99: {np.percentile(action_diffs, 99):.1f}")  # ~90
  # 设置阈值 = P99 × 1.1 ≈ 100
  ```

---

#### 维度 3：适用性 (Fitness for Purpose)

**ISO/IEC 5259 定义**：
> 数据是否适合特定的下游任务（训练、评估、部署）

**映射到 LeRobot**：
```
适用性问题（针对模仿学习训练）：
├── 图像质量
│   ├── 图像过暗/过亮（无法提取特征）
│   ├── 图像模糊（运动模糊、失焦）
│   ├── 图像方差过低（全黑/全白）
│   └── 遮挡严重（关键物体被遮挡）
│
├── 任务相关性
│   ├── 任务标签缺失
│   ├── 任务标签错误（pushT 但实际是 pick）
│   └── 演示质量差（人类失误、多次尝试）
│
└── 训练适用性
    ├── Episode 分布不均（某些状态过采样）
    ├── 动作空间覆盖不全（只在角落活动）
    └── 成功率过低（全是失败案例）
```

**具体规则**：

| 规则 ID | 检查项 | 阈值 | 代码示例 |
|---------|--------|------|---------|
| `image_low_variance` | 图像方差 | > 0.01 | `image.var() < 0.01` |
| `image_brightness` | 亮度异常 | [0.1, 0.9] | `image.mean() < 0.1 or > 0.9` |
| `task_label_missing` | 任务标签 | 必须存在 | `frame['task'] is None` |

**为什么方差 < 0.01？**
- **正常图像**：
  - 有物体、背景、纹理 → 像素值变化大 → 方差 ~0.05-0.15
  
- **异常图像**：
  - 全黑：所有像素 = 0 → 方差 = 0
  - 全白：所有像素 = 1 → 方差 = 0
  - 近乎均匀：方差 < 0.01
  
- **影响**：
  - CNN 需要从图像中提取特征（边缘、纹理、物体）
  - 方差过低 → 没有特征可提取 → 训练失败

---

#### 维度 4：可度量性 (Measurability)

**ISO/IEC 5259 定义**：
> 数据的数值是否可靠、有意义、在合理范围内

**映射到 LeRobot**：
```
可度量性问题：
├── 数值范围异常
│   ├── 状态超出工作空间（x > 500, y < 0）
│   ├── 动作超出物理限制（关节角度 > π）
│   └── 图像像素值异常（< 0 或 > 1）
│
├── 单位/量纲错误
│   ├── 角度用度而非弧度（180 vs π）
│   ├── 位置用米而非像素
│   └── 时间戳用毫秒而非秒
│
└── 数值分布异常
    ├── 状态/动作分布退化（只在一个点）
    ├── 异常值/离群点（3σ 之外）
    └── 精度丢失（浮点数溢出）
```

**具体规则**：

| 规则 ID | 检查项 | 阈值 | 代码示例 |
|---------|--------|------|---------|
| `state_out_of_range` | 状态范围 | [0, 500] | `state.min() < 0 or state.max() > 500` |
| `action_out_of_range` | 动作范围 | [0, 500] | `action.min() < 0 or action.max() > 500` |
| `image_out_of_range` | 图像范围 | [0, 1] | `image.min() < 0 or image.max() > 1.1` |

**为什么 [0, 500]？**
- **从数据中学习**：
  ```python
  # 统计所有状态的范围
  all_states = []
  # ... 收集
  print(f"Min: {np.min(all_states)}")  # 30.2
  print(f"Max: {np.max(all_states)}")  # 447.2
  
  # 加 10% 容忍度
  range_min = 30 * 0.9 ≈ 0
  range_max = 450 * 1.1 ≈ 500
  ```

- **物理意义**：
  - pusht 的工作空间是一个平面
  - 实际坐标范围约 [30, 450]
  - 设置 [0, 500] 作为"物理可能"的边界

---

## 四、完整的规则定义框架

### 4.1 规则定义模板

```python
RULE_TEMPLATE = {
    'rule_id': 'unique_identifier',
    'name': '规则名称',
    'dimension': 'Completeness | Consistency | Fitness | Measurability',
    'level': 'Episode | Frame',
    'threshold': '数值阈值（如果适用）',
    'description': '一句话描述',
    'rationale': '为什么这条规则重要',
    'impact': '违反规则会导致什么问题',
    'check_function': 'Python 函数',
    'severity': 'Critical | High | Medium | Low'
}
```

### 4.2 pusht 数据集的完整规则表

| 规则 ID | 维度 | 级别 | 阈值 | 严重性 | 理由 |
|---------|------|------|------|--------|------|
| `missing_fields` | Completeness | Frame | 0 | Critical | 训练代码会崩溃 |
| `nan_values` | Completeness | Frame | 0 | Critical | 导致梯度爆炸 |
| `too_short` | Completeness | Episode | ≥30 | High | 无法学到完整轨迹 |
| `timestamp_gap` | Consistency | Frame | \|dt-0.1\|<0.01 | High | 速度计算错误 |
| `action_jump` | Consistency | Frame | <100 px | High | 学到"瞬移"行为 |
| `image_low_var` | Fitness | Frame | >0.01 | Medium | CNN 无特征可提取 |
| `state_out_range` | Measurability | Frame | [0,500] | Medium | 物理上不可能 |

---

## 五、不同任务的规则调整

### 5.1 通用规则（所有任务）

**这些规则的阈值是 0，不需要调整**：
- `missing_fields`
- `nan_values`

---

### 5.2 需要调整的规则

| 规则 | pusht (2D 推方块) | aloha (双臂机器人) | pick_and_place |
|------|------------------|------------------|---------------|
| `too_short` | 30 帧 (3s) | 100 帧 (10s) | 50 帧 (5s) |
| `action_jump` | 100 像素 | 0.5 弧度（关节） | 50 mm（末端） |
| `state_range` | [0, 500] px | [-π, π] rad | [-1, 1] m |
| **新增规则** | - | 双臂碰撞检测 | 物体掉落检测 |

---

### 5.3 调整方法

**Step 1：继承通用规则**
```python
base_rules = {
    'missing_fields': {...},
    'nan_values': {...},
    # ... 完整性、一致性核心规则
}
```

**Step 2：覆盖任务特定阈值**
```python
aloha_rules = base_rules.copy()
aloha_rules['too_short']['threshold'] = 100  # 双臂任务更复杂
aloha_rules['action_jump']['threshold'] = 0.5  # 单位是弧度
aloha_rules['action_jump']['description'] = 'Joint angle change > 0.5 rad'
```

**Step 3：添加任务特定规则**
```python
aloha_rules['arm_collision'] = {
    'dimension': 'Fitness',
    'threshold': 0.05,  # 5cm
    'description': 'Two arms too close (< 5cm)',
    'check_function': check_arm_distance
}
```

---

## 六、规则验证与迭代

### 6.1 验证流程

```
1. 定义初版规则
   ↓
2. 应用到数据集
   ↓
3. 检查标记的 Episode
   ↓
4. 人工验证（是否真的是异常？）
   ↓
5. 计算指标：
   - 准确率 = 真阳性 / (真阳性 + 假阳性)
   - 召回率 = 真阳性 / (真阳性 + 假阴性)
   ↓
6. 调整阈值
   ↓
7. 重复 2-6
```

### 6.2 调整策略

**如果准确率低（很多误报）**：
```python
# 假设 action_jump 标记了 50% 的 Episode，但人工检查发现都是正常的
# → 阈值太严格

# 调整前
threshold = 100

# 查看被标记的 Episode 的实际动作变化
flagged_jumps = [120, 110, 105, 108, ...]  # 都在 100-120 之间

# 调整后
threshold = 150  # 放宽到 P99.5
```

**如果召回率低（很多漏报）**：
```python
# 假设人工检查发现有 10 个明显坏的 Episode 没被标记
# → 阈值太松或缺少规则

# 分析这 10 个 Episode 的特征
# 发现：它们的图像都很模糊，但方差检查没标记

# 添加新规则
rules['image_blurry'] = {
    'check': lambda img: compute_laplacian_variance(img) < threshold
}
```

---

## 七、面试准备

### 7.1 可以这样说

**Q: 你如何定义数据质量规则？**

> "我采用 ISO/IEC 5259 AI/ML 数据质量标准作为框架，将抽象的质量维度映射到 LeRobot 的具体数据结构。
> 
> **完整性维度**：检查 Episode 是否过短（< 30 帧）、字段是否缺失、是否有 NaN 值。30 帧的阈值是根据 pusht 任务的理论最短时间（3 秒）乘以 FPS（10）得出的。
> 
> **一致性维度**：检查时间戳是否连续、动作是否突变。动作突变阈值 100 像素是基于人类操作的物理限制——在 0.1 秒内不可能移动超过 100 像素，超过说明是数据错误或时序错位。
> 
> **适用性维度**：检查图像方差是否过低（< 0.01）。方差低说明图像几乎全黑或全白，CNN 无法提取特征，不适合训练。
> 
> **可度量性维度**：检查状态和动作是否在物理可能的范围内（[0, 500]）。这个范围是从数据统计中学习的，加上 10% 容忍度。
> 
> 这套框架的优势是可复用：对于不同任务（如 aloha 双臂机器人），只需要调整阈值和添加任务特定规则，核心框架保持不变。"

---

### 7.2 关键要点

**面试官可能追问的点**：

1. **"ISO/IEC 5259 具体是什么？"**
   - AI/ML 数据质量国际标准
   - 定义了完整性、一致性、适用性、可度量性等维度
   - 提供了数据质量评估的通用框架

2. **"为什么不直接用现成的数据质量工具（如 Great Expectations）？"**
   - 通用工具针对表格数据（SQL、CSV）
   - 机器人数据是多模态时序数据（图像 + 状态 + 动作 + 时间）
   - 需要领域特定的规则（动作突变、时序一致性）

3. **"阈值是怎么定的？不会太主观吗？"**
   - 结合三种方法：
     1. 数据驱动（P95、P99 分位数）
     2. 物理约束（速度限制、工作空间）
     3. 下游任务需求（训练收敛所需的最短轨迹）
   - 初版后通过人工验证和迭代优化

4. **"如何验证规则的有效性？"**
   - 人工抽样检查被标记的 Episode
   - 计算准确率和召回率
   - A/B 测试：对比清洗前后的训练效果

---

## 八、总结：从标准到实践的完整流程

```
ISO/IEC 5259 标准
    ↓
[四个维度]
  - Completeness (完整性)
  - Consistency (一致性)  
  - Fitness (适用性)
  - Measurability (可度量性)
    ↓
[映射到 LeRobot 数据结构]
  - Episode 级别：长度、任务、成功
  - Frame 级别：图像、状态、动作、时间戳
    ↓
[定义可执行规则]
  - too_short: Episode < 30 帧
  - action_jump: 相邻帧动作变化 > 100 像素
  - image_low_variance: 方差 < 0.01
  - value_out_of_range: 状态/动作超出 [0, 500]
    ↓
[确定阈值]
  - 数据驱动（统计分布）
  - 物理约束（人类速度限制）
  - 任务需求（完整轨迹长度）
    ↓
[验证与迭代]
  - 应用规则 → 人工检查 → 调整阈值
    ↓
[适配新任务]
  - 继承通用规则
  - 调整阈值
  - 添加任务特定规则
```

---

## 九、学习检查清单

- [x] 理解 ISO/IEC 5259 四个质量维度
- [x] 理解如何将抽象维度映射到具体数据结构
- [x] 理解每条规则的阈值来源（数据驱动 + 物理约束 + 任务需求）
- [x] 理解不同任务如何调整规则
- [x] 理解规则验证与迭代的流程
- [ ] 实际运行检测脚本，查看结果（下一步）
- [ ] 分析哪些规则有效，哪些需要调整（下一步）

---

**笔记更新日志**：
- 2026-09-10：创建笔记，详细解释从 ISO/IEC 5259 标准到具体规则的映射
