#!/usr/bin/env python3
"""
构建 PyTorch DataLoader

目标：
  1. 理解 LeRobot 的 DataLoader 结构
  2. 实现 Episode 级别随机采样
  3. 测试数据加载速度
  4. 可视化一个 batch 的数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import time
import matplotlib.pyplot as plt

print("=" * 70)
print("构建 PyTorch DataLoader")
print("=" * 70)

# 1. 查找数据文件
print("\n[步骤 1/6] 查找数据文件...")

cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "hub"
dataset_dir = None

for item in cache_dir.glob("*pusht*"):
    if item.is_dir():
        dataset_dir = item
        break

if not dataset_dir:
    print("❌ 未找到数据集")
    exit(1)

# 找到 snapshot
snapshots = list((dataset_dir / "snapshots").iterdir())
if not snapshots:
    print("❌ 未找到 snapshot")
    exit(1)

snapshot_dir = snapshots[0]
data_file = snapshot_dir / "data" / "chunk-000" / "file-000.parquet"

if not data_file.exists():
    print(f"❌ 未找到数据文件: {data_file}")
    exit(1)

print(f"✅ 找到数据文件: {data_file.name}")

# 2. 加载数据
print("\n[步骤 2/6] 加载 Parquet 数据...")

df = pd.read_parquet(data_file)
print(f"✅ 数据形状: {df.shape}")
print(f"   列: {list(df.columns)}")

# 3. 构建 Episode 索引
print("\n[步骤 3/6] 构建 Episode 索引...")

# 按 episode_index 分组
episode_groups = df.groupby('episode_index')

episode_info = []
for ep_idx, group in episode_groups:
    episode_info.append({
        'episode_index': ep_idx,
        'start_frame': group.index[0],  # 全局帧索引
        'end_frame': group.index[-1] + 1,
        'length': len(group)
    })

episode_info_df = pd.DataFrame(episode_info)
print(f"✅ Episode 数量: {len(episode_info_df)}")
print(f"\n前 5 个 Episode:")
print(episode_info_df.head())

# 4. 创建自定义 Dataset
print("\n[步骤 4/6] 创建 PyTorch Dataset...")

class PushTDataset(Dataset):
    def __init__(self, df, episode_info):
        self.df = df
        self.episode_info = episode_info

        # 为了演示，我们只使用数值数据（不加载视频）
        self.data = {
            'observation.state': np.stack(df['observation.state'].values),
            'action': np.stack(df['action'].values),
            'episode_index': df['episode_index'].values,
            'frame_index': df['frame_index'].values,
        }

        print(f"   数据集大小: {len(self.df)} 帧")
        print(f"   Episode 数量: {len(self.episode_info)}")

    def __len__(self):
        # 返回 Episode 数量（不是帧数！）
        return len(self.episode_info)

    def __getitem__(self, idx):
        """
        返回一个完整的 Episode
        """
        # 获取 Episode 信息
        ep_info = self.episode_info.iloc[idx]
        start = ep_info['start_frame']
        end = ep_info['end_frame']

        # 提取这个 Episode 的所有数据
        episode_data = {
            'observation.state': torch.FloatTensor(self.data['observation.state'][start:end]),
            'action': torch.FloatTensor(self.data['action'][start:end]),
            'episode_index': ep_info['episode_index'],
            'length': ep_info['length']
        }

        return episode_data

# 创建数据集
dataset = PushTDataset(df, episode_info_df)
print(f"✅ Dataset 创建完成")

# 5. 创建 DataLoader
print("\n[步骤 5/6] 创建 DataLoader...")

# 自定义 collate 函数（因为每个 Episode 长度不同）
def collate_fn(batch):
    """
    batch: list of episode_data
    """
    return batch  # 直接返回列表，因为 Episode 长度不同，无法堆叠

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,  # 每次加载 4 个 Episode
    shuffle=True,  # ← 这里实现了 Episode 级别随机！
    num_workers=0,  # 单进程（方便调试）
    collate_fn=collate_fn
)

print(f"✅ DataLoader 创建完成")
print(f"   Batch size: 4 Episodes")
print(f"   Shuffle: True (Episode 级别随机)")
print(f"   Total batches: {len(dataloader)}")

# 6. 测试 DataLoader
print("\n[步骤 6/6] 测试数据加载...")

# 加载一个 batch
start_time = time.time()
batch = next(iter(dataloader))
load_time = time.time() - start_time

print(f"\n📊 加载一个 Batch (4 Episodes):")
print(f"   耗时: {load_time:.4f} 秒")

for i, episode in enumerate(batch):
    print(f"\n   Episode {i+1}:")
    print(f"      episode_index: {episode['episode_index']}")
    print(f"      length: {episode['length']} 帧")
    print(f"      observation.state shape: {episode['observation.state'].shape}")
    print(f"      action shape: {episode['action'].shape}")
    print(f"      首帧状态: {episode['observation.state'][0].numpy()}")
    print(f"      首帧动作: {episode['action'][0].numpy()}")

# 7. 可视化一个 Episode 的轨迹
print("\n[步骤 7/7] 可视化 Episode 轨迹...")

episode = batch[0]  # 取第一个 Episode

states = episode['observation.state'].numpy()  # [length, 2]
actions = episode['action'].numpy()  # [length, 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图 1: 状态轨迹
axes[0].plot(states[:, 0], states[:, 1], 'b-', alpha=0.6, linewidth=2, label='State trajectory')
axes[0].scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
axes[0].scatter(states[-1, 0], states[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
axes[0].set_xlabel('X position')
axes[0].set_ylabel('Y position')
axes[0].set_title(f'State Trajectory (Episode {episode["episode_index"]}, {episode["length"]} frames)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(0, 500)
axes[0].invert_yaxis()  # 图像坐标系：Y 轴向下

# 图 2: 动作轨迹
axes[1].plot(actions[:, 0], actions[:, 1], 'r-', alpha=0.6, linewidth=2, label='Action trajectory')
axes[1].scatter(actions[0, 0], actions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
axes[1].scatter(actions[-1, 0], actions[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
axes[1].set_xlabel('X position')
axes[1].set_ylabel('Y position')
axes[1].set_title(f'Action Trajectory (Episode {episode["episode_index"]})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 500)
axes[1].set_ylim(0, 500)
axes[1].invert_yaxis()

plt.tight_layout()
output_path = "my_project_notes/episode_trajectory.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存: {output_path}")

# 8. 测试迭代速度
print("\n[步骤 8/8] 测试迭代速度...")

start_time = time.time()
num_batches = 10

for i, batch in enumerate(dataloader):
    if i >= num_batches:
        break

total_time = time.time() - start_time
print(f"\n📊 迭代 {num_batches} 个 batches:")
print(f"   总耗时: {total_time:.4f} 秒")
print(f"   平均每个 batch: {total_time / num_batches:.4f} 秒")
print(f"   平均每个 Episode: {total_time / (num_batches * 4):.4f} 秒")

print("\n" + "=" * 70)
print("✅ DataLoader 测试完成！")
print("=" * 70)

print("\n💡 关键发现:")
print("   1. DataLoader 的 shuffle=True 实现了 Episode 级别随机")
print("   2. 每个 batch 包含 4 个 Episode（长度可能不同）")
print("   3. Episode 内部的帧是顺序访问的（如果加载视频会很快）")
print("   4. 可以通过 num_workers 参数启用多进程并行加载")
