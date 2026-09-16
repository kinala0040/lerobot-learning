#!/usr/bin/env python3
"""
LeRobot 数据异常检测与清洗

学习目标：
1. 定义数据异常检测规则
2. 实现自动化异常检测
3. 生成异常报告
4. 对比清洗前后的数据统计

异常检测规则：
- Rule 1: Episode 过短（< 30 帧）
- Rule 2: 动作突变（相邻帧变化 > 100 像素）
- Rule 3: 图像异常（全黑、全白、低方差）
- Rule 4: 数值范围异常（超出合理范围）
"""

import torch
from lerobot.datasets import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

print("=" * 70)
print("LeRobot 数据异常检测")
print("=" * 70)

# 1. 加载数据集
print("\n[步骤 1/7] 加载数据集...")
dataset = LeRobotDataset('lerobot/pusht')
print(f"✅ 数据集: {dataset.repo_id}")
print(f"   总帧数: {len(dataset)}")
print(f"   Episode 数量: {dataset.num_episodes}")

# 2. 定义异常检测规则（基于 ISO/IEC 5259 + LeRobot 结构）
print("\n[步骤 2/7] 定义异常检测规则...")
print("📋 规则框架：ISO/IEC 5259 AI/ML 数据质量标准")
print("   参考：https://www.iso.org/standard/81088.html")
print("   应用于：LeRobot 多模态时序轨迹数据\n")

RULES = {
    # === 维度 1: 完整性 (Completeness) ===
    'too_short': {
        'name': 'Episode too short',
        'dimension': 'Completeness',
        'threshold': 30,  # 帧数
        'description': 'Episode < 30 frames (3s) - incomplete trajectory',
        'rationale': 'Cannot learn full task execution from too short episodes'
    },

    # === 维度 2: 一致性 (Consistency) ===
    'action_jump': {
        'name': 'Action jump',
        'dimension': 'Consistency',
        'threshold': 50,  # 像素（调整：原来 100，基于 P99.5 = 47.1）
        'description': 'Adjacent frame action change > 50 pixels',
        'rationale': 'Human cannot move 50+ pixels in 0.1s - likely data misalignment'
    },

    # === 维度 3: 适用性 (Fitness for Purpose) ===
    'image_anomaly': {
        'name': 'Image anomaly',
        'dimension': 'Fitness',
        'threshold': 0.0065,  # 方差阈值（调整：原来 0.01，基于 P1 = 0.00654）
        'description': 'Image variance < 0.0065 (too uniform)',
        'rationale': 'Nearly uniform image - CNN cannot extract features'
    },

    # === 维度 4: 可度量性 (Measurability) ===
    'value_out_of_range': {
        'name': 'Value out of range',
        'dimension': 'Measurability',
        'description': 'State or action outside expected range [0, 500]',
        'rationale': 'Values beyond physical workspace - sensor error or coordinate transform issue'
    }
}

print(f"定义了 {len(RULES)} 条检测规则，按 ISO/IEC 5259 四个维度分类:")
for dimension in ['Completeness', 'Consistency', 'Fitness', 'Measurability']:
    rules_in_dim = [r for r in RULES.values() if r.get('dimension') == dimension]
    print(f"\n   [{dimension}]")
    for rule in rules_in_dim:
        print(f"      - {rule['name']}: {rule['description']}")
        print(f"        理由: {rule['rationale']}")

# 3. 遍历所有 Episode 进行检测
print("\n[步骤 3/7] 检测异常 Episode...")
anomalies = defaultdict(list)  # {rule_id: [episode_indices]}
episode_details = []  # 每个 Episode 的详细信息

for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx

    ep_detail = {
        'episode': ep_idx,
        'length': length,
        'anomalies': []
    }

    # Rule 1: Episode 过短
    if length < RULES['too_short']['threshold']:
        anomalies['too_short'].append(ep_idx)
        ep_detail['anomalies'].append('too_short')

    # Rule 2: 动作突变检测
    actions = []
    for i in range(from_idx, to_idx):
        frame = dataset[i]
        actions.append(frame['action'].numpy())

    actions = np.array(actions)
    action_diffs = np.linalg.norm(actions[1:] - actions[:-1], axis=1)  # L2 距离
    max_action_jump = action_diffs.max() if len(action_diffs) > 0 else 0

    if max_action_jump > RULES['action_jump']['threshold']:
        anomalies['action_jump'].append(ep_idx)
        ep_detail['anomalies'].append('action_jump')
        ep_detail['max_action_jump'] = float(max_action_jump)

    # Rule 3: 图像异常检测（抽样检查前 3 帧）
    sample_frames = min(3, length)
    image_variances = []
    for i in range(from_idx, from_idx + sample_frames):
        frame = dataset[i]
        img = frame['observation.image']
        variance = img.var().item()
        image_variances.append(variance)

    min_variance = min(image_variances)
    if min_variance < RULES['image_anomaly']['threshold']:
        anomalies['image_anomaly'].append(ep_idx)
        ep_detail['anomalies'].append('image_anomaly')
        ep_detail['min_image_variance'] = float(min_variance)

    # Rule 4: 数值范围检测
    states = []
    for i in range(from_idx, to_idx):
        frame = dataset[i]
        states.append(frame['observation.state'].numpy())

    states = np.array(states)
    if states.min() < 0 or states.max() > 500:
        anomalies['value_out_of_range'].append(ep_idx)
        ep_detail['anomalies'].append('value_out_of_range')

    episode_details.append(ep_detail)

# 4. 统计异常情况
print(f"\n📊 异常检测结果:")
total_anomalous = len(set(sum([eps for eps in anomalies.values()], [])))
print(f"   异常 Episode 总数: {total_anomalous}/{dataset.num_episodes} "
      f"({total_anomalous/dataset.num_episodes*100:.1f}%)")
print(f"\n   按规则分类:")
for rule_id, eps in anomalies.items():
    rule_name = RULES[rule_id]['name']
    print(f"   - {rule_name}: {len(eps)} episodes")
    if len(eps) <= 5:
        print(f"       Episode IDs: {eps}")
    else:
        print(f"       Episode IDs: {eps[:5]} ... (showing first 5)")

# 5. 生成"干净"数据集的 Episode 列表
print("\n[步骤 4/7] 生成清洗后的 Episode 列表...")
anomalous_episodes = set(sum([eps for eps in anomalies.values()], []))
clean_episodes = [ep for ep in range(dataset.num_episodes)
                  if ep not in anomalous_episodes]

print(f"✅ 清洗后保留: {len(clean_episodes)}/{dataset.num_episodes} episodes "
      f"({len(clean_episodes)/dataset.num_episodes*100:.1f}%)")

# 6. 对比清洗前后的统计
print("\n[步骤 5/7] 对比清洗前后的统计...")
all_lengths = []
clean_lengths = []

for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx
    all_lengths.append(length)
    if ep_idx in clean_episodes:
        clean_lengths.append(length)

all_lengths = np.array(all_lengths)
clean_lengths = np.array(clean_lengths)

print(f"\n📊 清洗前:")
print(f"   Episode 数量: {len(all_lengths)}")
print(f"   平均长度: {all_lengths.mean():.1f} 帧")
print(f"   标准差: {all_lengths.std():.1f} 帧")
print(f"   最短: {all_lengths.min()} 帧")

print(f"\n📊 清洗后:")
print(f"   Episode 数量: {len(clean_lengths)}")
print(f"   平均长度: {clean_lengths.mean():.1f} 帧")
print(f"   标准差: {clean_lengths.std():.1f} 帧")
print(f"   最短: {clean_lengths.min()} 帧")

print(f"\n📊 提升:")
print(f"   平均长度提升: {clean_lengths.mean() - all_lengths.mean():.1f} 帧 "
      f"({(clean_lengths.mean() - all_lengths.mean()) / all_lengths.mean() * 100:.1f}%)")
print(f"   标准差改善: {all_lengths.std() - clean_lengths.std():.1f} 帧")

# 7. 生成可视化报告
print("\n[步骤 6/7] 生成可视化报告...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 清洗前后长度分布对比
axes[0, 0].hist([all_lengths, clean_lengths], bins=30,
                label=['Before cleaning', 'After cleaning'],
                edgecolor='black', alpha=0.7)
axes[0, 0].axvline(all_lengths.mean(), color='blue', linestyle='--', alpha=0.7,
                   label=f'Before mean: {all_lengths.mean():.1f}')
axes[0, 0].axvline(clean_lengths.mean(), color='orange', linestyle='--', alpha=0.7,
                   label=f'After mean: {clean_lengths.mean():.1f}')
axes[0, 0].set_xlabel('Episode Length (frames)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Episode Length Distribution: Before vs After Cleaning')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 异常类型分布
rule_names = [RULES[rid]['name'] for rid in anomalies.keys()]
rule_counts = [len(eps) for eps in anomalies.values()]
axes[0, 1].bar(range(len(rule_names)), rule_counts, edgecolor='black', alpha=0.7)
axes[0, 1].set_xticks(range(len(rule_names)))
axes[0, 1].set_xticklabels(rule_names, rotation=45, ha='right')
axes[0, 1].set_ylabel('Number of Episodes')
axes[0, 1].set_title('Anomaly Types Distribution')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 图3: Episode 长度散点图（标记异常）
all_ep_indices = np.arange(dataset.num_episodes)
colors = ['red' if ep in anomalous_episodes else 'blue'
          for ep in all_ep_indices]
axes[1, 0].scatter(all_ep_indices, all_lengths, c=colors, alpha=0.6, s=20)
axes[1, 0].axhline(RULES['too_short']['threshold'], color='red',
                   linestyle='--', label='Too short threshold')
axes[1, 0].set_xlabel('Episode Index')
axes[1, 0].set_ylabel('Episode Length (frames)')
axes[1, 0].set_title('Episode Length by Index (Red = Anomalous)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 图4: 数据保留率
labels = ['Cleaned\n(Removed)', f'Kept\n({len(clean_episodes)})']
sizes = [total_anomalous, len(clean_episodes)]
colors_pie = ['lightcoral', 'lightblue']
axes[1, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
axes[1, 1].set_title(f'Data Retention Rate\n({len(clean_episodes)}/{dataset.num_episodes} episodes)')

plt.tight_layout()
output_path = "/home/kinala/lerebot/lerobot/my_project_notes/anomaly_detection_report.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 可视化报告已保存: {output_path}")

# 8. 保存异常检测报告（JSON）
print("\n[步骤 7/7] 保存异常检测报告...")
report = {
    'dataset': dataset.repo_id,
    'total_episodes': dataset.num_episodes,
    'detection_rules': {rid: {
        'name': r['name'],
        'description': r['description'],
        'threshold': r.get('threshold', 'N/A')
    } for rid, r in RULES.items()},
    'anomaly_summary': {
        'total_anomalous': total_anomalous,
        'anomalous_rate': total_anomalous / dataset.num_episodes,
        'by_rule': {rid: len(eps) for rid, eps in anomalies.items()}
    },
    'anomalous_episodes': list(anomalous_episodes),
    'clean_episodes': clean_episodes,
    'statistics': {
        'before_cleaning': {
            'episode_count': len(all_lengths),
            'mean_length': float(all_lengths.mean()),
            'std_length': float(all_lengths.std()),
            'min_length': int(all_lengths.min()),
            'max_length': int(all_lengths.max())
        },
        'after_cleaning': {
            'episode_count': len(clean_lengths),
            'mean_length': float(clean_lengths.mean()),
            'std_length': float(clean_lengths.std()),
            'min_length': int(clean_lengths.min()),
            'max_length': int(clean_lengths.max())
        },
        'improvement': {
            'mean_length_increase': float(clean_lengths.mean() - all_lengths.mean()),
            'std_reduction': float(all_lengths.std() - clean_lengths.std())
        }
    },
    'episode_details': episode_details[:10]  # 只保存前 10 个作为示例
}

json_path = "/home/kinala/lerebot/lerobot/my_project_notes/anomaly_detection_report.json"
with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✅ JSON 报告已保存: {json_path}")

# 9. 保存清洗后的 Episode 列表
clean_list_path = "/home/kinala/lerebot/lerobot/my_project_notes/clean_episodes.txt"
with open(clean_list_path, 'w') as f:
    f.write(f"# Clean episodes for {dataset.repo_id}\n")
    f.write(f"# Total: {len(clean_episodes)}/{dataset.num_episodes}\n")
    f.write(f"# Generated: 2026-09-10\n\n")
    for ep in clean_episodes:
        f.write(f"{ep}\n")
print(f"✅ 清洗后 Episode 列表已保存: {clean_list_path}")

print("\n" + "=" * 70)
print("✅ 异常检测完成！")
print("=" * 70)
print(f"\n📄 生成的文件:")
print(f"   - 可视化: {output_path}")
print(f"   - JSON 报告: {json_path}")
print(f"   - 清洗列表: {clean_list_path}")
print(f"\n💡 关键发现:")
print(f"   - 异常 Episode: {total_anomalous}/{dataset.num_episodes} "
      f"({total_anomalous/dataset.num_episodes*100:.1f}%)")
print(f"   - 数据保留率: {len(clean_episodes)/dataset.num_episodes*100:.1f}%")
print(f"   - 平均长度提升: {clean_lengths.mean() - all_lengths.mean():.1f} 帧")
print(f"   - 标准差改善: {all_lengths.std() - clean_lengths.std():.1f} 帧")
