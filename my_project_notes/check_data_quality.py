#!/usr/bin/env python3
"""
LeRobot 数据质量检查脚本

学习目标：
1. 理解机器人数据可能有哪些质量问题
2. 学会编写自动化检查脚本
3. 生成数据质量报告

检查项目：
- Episode 长度统计
- 时间戳连续性
- 数据完整性
- 数值范围
- 成功率统计
"""

import torch
from lerobot.datasets import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

print("=" * 70)
print("LeRobot 数据质量检查")
print("=" * 70)

# 1. 加载数据集
print("\n[步骤 1/6] 加载数据集...")
dataset = LeRobotDataset('lerobot/pusht')
print(f"✅ 数据集: {dataset.repo_id}")
print(f"   总帧数: {len(dataset)}")
print(f"   Episode 数量: {dataset.num_episodes}")
print(f"   FPS: {dataset.fps}")

# 2. Episode 长度统计
print("\n[步骤 2/6] 分析 Episode 长度分布...")
episode_lengths = []
episode_success = []
episode_timestamps = []

for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx
    episode_lengths.append(length)

    # 检查最后一帧是否成功
    last_frame = dataset[to_idx - 1]
    success = last_frame['next.success'].item()
    episode_success.append(success)

    # 收集时间戳
    timestamps = [dataset[i]['timestamp'].item() for i in range(from_idx, to_idx)]
    episode_timestamps.append(timestamps)

episode_lengths = np.array(episode_lengths)
episode_success = np.array(episode_success)

print(f"\n📊 Episode 长度统计:")
print(f"   最短: {episode_lengths.min()} 帧")
print(f"   最长: {episode_lengths.max()} 帧")
print(f"   平均: {episode_lengths.mean():.2f} 帧")
print(f"   中位数: {np.median(episode_lengths):.0f} 帧")
print(f"   标准差: {episode_lengths.std():.2f} 帧")

print(f"\n📊 任务成功率:")
success_rate = episode_success.mean() * 100
print(f"   成功 Episode: {episode_success.sum()}/{len(episode_success)}")
print(f"   成功率: {success_rate:.2f}%")

# 3. 时间戳连续性检查
print("\n[步骤 3/6] 检查时间戳连续性...")
expected_dt = 1.0 / dataset.fps  # 应该是 0.1 秒
timestamp_issues = []

for ep_idx in range(dataset.num_episodes):
    timestamps = episode_timestamps[ep_idx]
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        # 允许 5% 的误差
        if abs(dt - expected_dt) > expected_dt * 0.05:
            timestamp_issues.append({
                'episode': ep_idx,
                'frame': i,
                'expected_dt': expected_dt,
                'actual_dt': dt,
                'error': abs(dt - expected_dt)
            })

if len(timestamp_issues) == 0:
    print(f"✅ 时间戳检查通过：所有时间戳连续且均匀")
else:
    print(f"⚠️  发现 {len(timestamp_issues)} 个时间戳异常")
    print(f"   显示前 5 个:")
    for issue in timestamp_issues[:5]:
        print(f"   - Episode {issue['episode']}, Frame {issue['frame']}: "
              f"dt={issue['actual_dt']:.4f}s (期望 {issue['expected_dt']:.4f}s)")

# 4. 数据完整性检查
print("\n[步骤 4/6] 检查数据完整性...")
missing_fields = []
nan_values = []

# 检查前 100 帧（抽样检查）
sample_indices = np.linspace(0, len(dataset)-1, 100, dtype=int)

for idx in sample_indices:
    frame = dataset[idx]

    # 检查必需字段
    required_fields = ['observation.image', 'observation.state', 'action', 'timestamp']
    for field in required_fields:
        if field not in frame:
            missing_fields.append((idx, field))
        else:
            # 检查 NaN
            if torch.is_tensor(frame[field]) and torch.isnan(frame[field]).any():
                nan_values.append((idx, field))

if len(missing_fields) == 0 and len(nan_values) == 0:
    print(f"✅ 数据完整性检查通过：所有必需字段存在且无 NaN")
else:
    if len(missing_fields) > 0:
        print(f"⚠️  发现 {len(missing_fields)} 个缺失字段")
    if len(nan_values) > 0:
        print(f"⚠️  发现 {len(nan_values)} 个 NaN 值")

# 5. 数值范围检查
print("\n[步骤 5/6] 检查数值范围...")
image_values = []
state_values = []
action_values = []

for idx in sample_indices:
    frame = dataset[idx]
    image_values.append((frame['observation.image'].min().item(),
                        frame['observation.image'].max().item()))
    state_values.append(frame['observation.state'].numpy())
    action_values.append(frame['action'].numpy())

image_min = min([v[0] for v in image_values])
image_max = max([v[1] for v in image_values])
state_values = np.array(state_values)
action_values = np.array(action_values)

print(f"\n📊 数值范围:")
print(f"   图像 (observation.image):")
print(f"      范围: [{image_min:.3f}, {image_max:.3f}]")
print(f"      期望: [0.0, 1.0] (归一化)")
if image_min < 0 or image_max > 1.01:
    print(f"      ⚠️  图像数值超出预期范围")

print(f"\n   状态 (observation.state):")
print(f"      X 范围: [{state_values[:, 0].min():.1f}, {state_values[:, 0].max():.1f}]")
print(f"      Y 范围: [{state_values[:, 1].min():.1f}, {state_values[:, 1].max():.1f}]")

print(f"\n   动作 (action):")
print(f"      X 范围: [{action_values[:, 0].min():.1f}, {action_values[:, 0].max():.1f}]")
print(f"      Y 范围: [{action_values[:, 1].min():.1f}, {action_values[:, 1].max():.1f}]")

# 6. 生成可视化报告
print("\n[步骤 6/6] 生成可视化报告...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1: Episode 长度分布
axes[0, 0].hist(episode_lengths, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(episode_lengths.mean(), color='red', linestyle='--',
                   label=f'Mean: {episode_lengths.mean():.1f}')
axes[0, 0].set_xlabel('Episode Length (frames)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Episode Length Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 成功 vs 失败的长度对比
success_lengths = episode_lengths[episode_success == 1]
failure_lengths = episode_lengths[episode_success == 0]
axes[0, 1].hist([success_lengths, failure_lengths], bins=20,
                label=['Success', 'Failure'], edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Episode Length (frames)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Success vs Failure (Success Rate: {success_rate:.1f}%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 图3: 状态空间分布
axes[1, 0].scatter(state_values[:, 0], state_values[:, 1], alpha=0.5, s=10)
axes[1, 0].set_xlabel('X Coordinate')
axes[1, 0].set_ylabel('Y Coordinate')
axes[1, 0].set_title('Robot State Space Distribution')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 500)
axes[1, 0].set_ylim(0, 500)

# 图4: 动作空间分布
axes[1, 1].scatter(action_values[:, 0], action_values[:, 1], alpha=0.5, s=10, color='orange')
axes[1, 1].set_xlabel('Action X')
axes[1, 1].set_ylabel('Action Y')
axes[1, 1].set_title('Action Space Distribution')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, 500)
axes[1, 1].set_ylim(0, 500)

plt.tight_layout()
output_path = "/home/kinala/lerebot/lerobot/my_project_notes/data_quality_report.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 可视化报告已保存: {output_path}")

# 7. 保存 JSON 报告
print("\n[步骤 7/7] 保存 JSON 报告...")
report = {
    'dataset': dataset.repo_id,
    'total_frames': len(dataset),
    'total_episodes': dataset.num_episodes,
    'fps': dataset.fps,
    'episode_length': {
        'min': int(episode_lengths.min()),
        'max': int(episode_lengths.max()),
        'mean': float(episode_lengths.mean()),
        'median': float(np.median(episode_lengths)),
        'std': float(episode_lengths.std())
    },
    'success_rate': {
        'success_count': int(episode_success.sum()),
        'total_count': int(len(episode_success)),
        'rate': float(success_rate)
    },
    'timestamp_issues': len(timestamp_issues),
    'data_integrity': {
        'missing_fields': len(missing_fields),
        'nan_values': len(nan_values)
    },
    'value_ranges': {
        'image': {'min': float(image_min), 'max': float(image_max)},
        'state_x': {'min': float(state_values[:, 0].min()), 'max': float(state_values[:, 0].max())},
        'state_y': {'min': float(state_values[:, 1].min()), 'max': float(state_values[:, 1].max())},
        'action_x': {'min': float(action_values[:, 0].min()), 'max': float(action_values[:, 0].max())},
        'action_y': {'min': float(action_values[:, 1].min()), 'max': float(action_values[:, 1].max())}
    }
}

json_path = "/home/kinala/lerebot/lerobot/my_project_notes/data_quality_report.json"
with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✅ JSON 报告已保存: {json_path}")

print("\n" + "=" * 70)
print("✅ 数据质量检查完成！")
print("=" * 70)
print(f"\n📄 报告文件:")
print(f"   - 可视化: {output_path}")
print(f"   - JSON: {json_path}")
print(f"\n💡 主要发现:")
print(f"   - Episode 数量: {dataset.num_episodes}")
print(f"   - 平均长度: {episode_lengths.mean():.1f} 帧 ({episode_lengths.mean()/dataset.fps:.1f} 秒)")
print(f"   - 成功率: {success_rate:.1f}%")
print(f"   - 时间戳异常: {len(timestamp_issues)} 个")
print(f"   - 数据完整性: {'✅ 通过' if len(missing_fields)==0 and len(nan_values)==0 else '⚠️ 有问题'}")
