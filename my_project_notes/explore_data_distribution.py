#!/usr/bin/env python3
"""
数据探索：统计真实的数值分布，用于设置合理的阈值

目标：找到图像方差、动作变化、数值范围的实际分布
"""

import torch
from lerobot.datasets import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LeRobot 数据分布探索")
print("=" * 70)

# 1. 加载数据集
print("\n[1/4] 加载数据集...")
dataset = LeRobotDataset('lerobot/pusht')
print(f"✅ 数据集: {dataset.repo_id}")

# 2. 统计图像方差分布
print("\n[2/4] 统计图像方差分布...")
image_variances = []

# 每个 Episode 采样前 3 帧
for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]

    sample_size = min(3, to_idx - from_idx)
    for i in range(from_idx, from_idx + sample_size):
        frame = dataset[i]
        img = frame['observation.image']
        variance = img.var().item()
        image_variances.append(variance)

image_variances = np.array(image_variances)

print(f"\n📊 图像方差统计:")
print(f"   最小值: {image_variances.min():.6f}")
print(f"   最大值: {image_variances.max():.6f}")
print(f"   平均值: {image_variances.mean():.6f}")
print(f"   中位数: {np.median(image_variances):.6f}")
print(f"   P1:  {np.percentile(image_variances, 1):.6f}")
print(f"   P5:  {np.percentile(image_variances, 5):.6f}")
print(f"   P10: {np.percentile(image_variances, 10):.6f}")

print(f"\n💡 建议阈值:")
print(f"   - 严格 (P1):  variance < {np.percentile(image_variances, 1):.6f}")
print(f"   - 中等 (P5):  variance < {np.percentile(image_variances, 5):.6f}")
print(f"   - 宽松 (P10): variance < {np.percentile(image_variances, 10):.6f}")

# 3. 统计动作变化分布
print("\n[3/4] 统计动作变化分布...")
action_changes = []

for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]

    actions = []
    for i in range(from_idx, to_idx):
        frame = dataset[i]
        actions.append(frame['action'].numpy())

    actions = np.array(actions)
    if len(actions) > 1:
        diffs = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
        action_changes.extend(diffs)

action_changes = np.array(action_changes)

print(f"\n📊 动作变化统计:")
print(f"   最小值: {action_changes.min():.2f} 像素")
print(f"   最大值: {action_changes.max():.2f} 像素")
print(f"   平均值: {action_changes.mean():.2f} 像素")
print(f"   中位数: {np.median(action_changes):.2f} 像素")
print(f"   P95: {np.percentile(action_changes, 95):.2f} 像素")
print(f"   P99: {np.percentile(action_changes, 99):.2f} 像素")
print(f"   P99.5: {np.percentile(action_changes, 99.5):.2f} 像素")

print(f"\n💡 建议阈值:")
print(f"   - 严格 (P95):  action_jump > {np.percentile(action_changes, 95):.2f} 像素")
print(f"   - 中等 (P99):  action_jump > {np.percentile(action_changes, 99):.2f} 像素")
print(f"   - 宽松 (P99.5): action_jump > {np.percentile(action_changes, 99.5):.2f} 像素")

# 4. 统计数值范围
print("\n[4/4] 统计状态和动作的数值范围...")
all_states = []
all_actions = []

sample_indices = np.linspace(0, len(dataset)-1, 500, dtype=int)
for idx in sample_indices:
    frame = dataset[idx]
    all_states.append(frame['observation.state'].numpy())
    all_actions.append(frame['action'].numpy())

all_states = np.array(all_states)
all_actions = np.array(all_actions)

print(f"\n📊 状态范围:")
print(f"   X: [{all_states[:, 0].min():.1f}, {all_states[:, 0].max():.1f}]")
print(f"   Y: [{all_states[:, 1].min():.1f}, {all_states[:, 1].max():.1f}]")

print(f"\n📊 动作范围:")
print(f"   X: [{all_actions[:, 0].min():.1f}, {all_actions[:, 0].max():.1f}]")
print(f"   Y: [{all_actions[:, 1].min():.1f}, {all_actions[:, 1].max():.1f}]")

print(f"\n💡 建议范围 (加 10% 容忍度):")
state_min = min(all_states.min(), all_actions.min()) * 0.9
state_max = max(all_states.max(), all_actions.max()) * 1.1
print(f"   状态和动作: [{state_min:.1f}, {state_max:.1f}]")

# 5. 可视化
print("\n[5/5] 生成可视化...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1: 图像方差分布
axes[0, 0].hist(image_variances, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0.01, color='red', linestyle='--', label='Current threshold: 0.01')
axes[0, 0].axvline(np.percentile(image_variances, 5), color='green', linestyle='--',
                   label=f'P5: {np.percentile(image_variances, 5):.4f}')
axes[0, 0].set_xlabel('Image Variance')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Image Variance Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 动作变化分布
axes[0, 1].hist(action_changes, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(100, color='red', linestyle='--', label='Current threshold: 100 px')
axes[0, 1].axvline(np.percentile(action_changes, 99), color='green', linestyle='--',
                   label=f'P99: {np.percentile(action_changes, 99):.1f} px')
axes[0, 1].set_xlabel('Action Change (pixels)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Action Change Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 图3: 图像方差 CDF（累积分布）
sorted_vars = np.sort(image_variances)
cdf = np.arange(1, len(sorted_vars)+1) / len(sorted_vars)
axes[1, 0].plot(sorted_vars, cdf, linewidth=2)
axes[1, 0].axvline(0.01, color='red', linestyle='--', label='Current: 0.01')
axes[1, 0].axhline(0.05, color='gray', linestyle=':', alpha=0.5)
axes[1, 0].axhline(0.10, color='gray', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Image Variance')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Image Variance CDF')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, np.percentile(image_variances, 99))

# 图4: 动作变化 CDF
sorted_actions = np.sort(action_changes)
cdf = np.arange(1, len(sorted_actions)+1) / len(sorted_actions)
axes[1, 1].plot(sorted_actions, cdf, linewidth=2)
axes[1, 1].axvline(100, color='red', linestyle='--', label='Current: 100 px')
axes[1, 1].axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='P95')
axes[1, 1].axhline(0.99, color='gray', linestyle=':', alpha=0.5, label='P99')
axes[1, 1].set_xlabel('Action Change (pixels)')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Action Change CDF')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = "/home/kinala/lerebot/lerobot/my_project_notes/data_distribution_exploration.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 可视化已保存: {output_path}")

print("\n" + "=" * 70)
print("✅ 数据探索完成！")
print("=" * 70)
print("\n💡 下一步：根据上面的统计结果，调整 detect_anomalies.py 中的阈值")
