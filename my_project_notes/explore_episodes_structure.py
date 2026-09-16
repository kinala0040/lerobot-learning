#!/usr/bin/env python3
"""
探索 dataset.meta.episodes 的数据结构

目标：理解 Episode 元数据的组织方式
"""

from lerobot.datasets import LeRobotDataset
import pandas as pd

print("=" * 70)
print("探索 dataset.meta.episodes 结构")
print("=" * 70)

# 加载数据集
dataset = LeRobotDataset('lerobot/pusht')

print(f"\n数据集基本信息:")
print(f"  总帧数: {len(dataset)}")
print(f"  Episode 数量: {dataset.num_episodes}")
print(f"  FPS: {dataset.fps}")

# 1. 查看 dataset.meta.episodes 的类型和结构
print(f"\n" + "=" * 70)
print("1. dataset.meta.episodes 的类型和结构")
print("=" * 70)
print(f"类型: {type(dataset.meta.episodes)}")
print(f"是否是 DataFrame: {isinstance(dataset.meta.episodes, pd.DataFrame)}")

# 2. 查看所有列名
print(f"\n" + "=" * 70)
print("2. 包含的列（字段）")
print("=" * 70)
# Hugging Face Dataset 使用 column_names 属性
print(f"列名: {dataset.meta.episodes.column_names}")

# 3. 查看前 10 个 Episode 的完整信息
print(f"\n" + "=" * 70)
print("3. 前 10 个 Episode 的详细信息")
print("=" * 70)
# Hugging Face Dataset 使用切片语法
print(dataset.meta.episodes[:10])

# 4. 解释 from_index 和 to_index
print(f"\n" + "=" * 70)
print("4. dataset_from_index 和 dataset_to_index 的含义")
print("=" * 70)

print("\n这两个字段定义了每个 Episode 在全局数据集中的帧范围：\n")

for ep_idx in range(min(5, dataset.num_episodes)):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx

    print(f"Episode {ep_idx}:")
    print(f"  dataset_from_index: {from_idx}")
    print(f"  dataset_to_index:   {to_idx}")
    print(f"  长度: {length} 帧")
    print(f"  含义: Episode {ep_idx} 占用全局数据集的第 {from_idx} - {to_idx-1} 帧")
    print(f"  访问方式: dataset[{from_idx}] 到 dataset[{to_idx-1}]")
    print()

# 5. 验证帧范围
print(f"\n" + "=" * 70)
print("5. 验证 Episode 帧范围是否连续")
print("=" * 70)

all_ranges = []
for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    all_ranges.append((from_idx, to_idx))

# 检查是否连续
is_continuous = True
for i in range(len(all_ranges) - 1):
    current_end = all_ranges[i][1]
    next_start = all_ranges[i+1][0]
    if current_end != next_start:
        print(f"⚠️  Episode {i} 和 Episode {i+1} 之间有间隙：")
        print(f"   Episode {i} 结束于 {current_end}")
        print(f"   Episode {i+1} 开始于 {next_start}")
        is_continuous = False

if is_continuous:
    print("✅ 所有 Episode 的帧范围是连续的")
    print(f"   Episode 0 从帧 {all_ranges[0][0]} 开始")
    print(f"   Episode {dataset.num_episodes-1} 在帧 {all_ranges[-1][1]} 结束")
    print(f"   总帧数: {all_ranges[-1][1]} (应该等于 {len(dataset)})")

# 6. 查看具体的一帧数据，确认 episode_index 和 frame_index
print(f"\n" + "=" * 70)
print("6. 查看具体帧数据，理解 episode_index 和 frame_index")
print("=" * 70)

# Episode 0 的第一帧
ep0_first = dataset[0]
print(f"\n全局帧 0 (Episode 0 的第一帧):")
print(f"  episode_index: {ep0_first['episode_index'].item()}")
print(f"  frame_index: {ep0_first['frame_index'].item()}")
print(f"  timestamp: {ep0_first['timestamp'].item():.3f}")

# Episode 0 的最后一帧
ep0_end = dataset.meta.episodes['dataset_to_index'][0] - 1
ep0_last = dataset[ep0_end]
print(f"\n全局帧 {ep0_end} (Episode 0 的最后一帧):")
print(f"  episode_index: {ep0_last['episode_index'].item()}")
print(f"  frame_index: {ep0_last['frame_index'].item()}")
print(f"  timestamp: {ep0_last['timestamp'].item():.3f}")

# Episode 1 的第一帧
ep1_first_idx = dataset.meta.episodes['dataset_from_index'][1]
ep1_first = dataset[ep1_first_idx]
print(f"\n全局帧 {ep1_first_idx} (Episode 1 的第一帧):")
print(f"  episode_index: {ep1_first['episode_index'].item()}")
print(f"  frame_index: {ep1_first['frame_index'].item()}")
print(f"  timestamp: {ep1_first['timestamp'].item():.3f}")

# 7. 总结
print(f"\n" + "=" * 70)
print("7. 总结")
print("=" * 70)

print("""
数据组织结构：

1. 全局数据集层：
   dataset[0], dataset[1], ..., dataset[25649]
   ↑ 所有 Episode 的所有帧按顺序排列

2. Episode 元数据层（dataset.meta.episodes）：
   一个 DataFrame，每行代表一个 Episode

   列：
   - dataset_from_index: Episode 在全局数据集中的起始帧索引
   - dataset_to_index: Episode 在全局数据集中的结束帧索引（不包含）
   - 其他可能的列：task, success, length 等

3. 访问方式：

   方式 1：通过全局索引访问
   frame = dataset[100]  # 访问全局第 100 帧

   方式 2：通过 Episode 索引访问
   ep_idx = 0
   from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
   to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]

   # 访问 Episode 0 的所有帧
   for i in range(from_idx, to_idx):
       frame = dataset[i]

4. 关键理解：
   - dataset_from_index 和 dataset_to_index 定义了 Episode 的"边界"
   - 范围是 [from_index, to_index)，左闭右开
   - Episode 0: [0, 161)   → 帧 0-160
   - Episode 1: [161, 284) → 帧 161-283
   - Episode 2: [284, ...) → ...

   所以 Episode 1 的第一帧就是全局的第 161 帧
""")

print("\n" + "=" * 70)
print("✅ 探索完成！")
print("=" * 70)
