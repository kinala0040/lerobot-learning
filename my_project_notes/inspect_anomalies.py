#!/usr/bin/env python3
"""
查看被标记为异常的 Episode 详情

目标：人工验证异常检测是否合理
"""

import json
import numpy as np
from lerobot.datasets import LeRobotDataset

print("=" * 70)
print("异常 Episode 详情查看")
print("=" * 70)

# 1. 读取检测报告
report_path = "/home/kinala/lerebot/lerobot/my_project_notes/anomaly_detection_report.json"
with open(report_path, 'r') as f:
    report = json.load(f)

print(f"\n数据集: {report['dataset']}")
print(f"总 Episode 数: {report['total_episodes']}")
print(f"异常 Episode 数: {report['anomaly_summary']['total_anomalous']}")

# 2. 加载数据集
dataset = LeRobotDataset('lerobot/pusht')

# 3. 查看动作突变的 Episode（最多 10 个）
action_jump_episodes = []
for ep_detail in report['episode_details']:
    if 'action_jump' in ep_detail['anomalies']:
        action_jump_episodes.append(ep_detail)

print(f"\n" + "=" * 70)
print(f"🔍 查看动作突变 Episode（前 10 个）")
print("=" * 70)

for i, ep_detail in enumerate(action_jump_episodes[:10]):
    ep_idx = ep_detail['episode']
    max_jump = ep_detail.get('max_action_jump', 'N/A')

    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx

    # 收集这个 Episode 的所有动作变化
    actions = []
    for j in range(from_idx, to_idx):
        frame = dataset[j]
        actions.append(frame['action'].numpy())

    actions = np.array(actions)
    action_diffs = np.linalg.norm(actions[1:] - actions[:-1], axis=1)

    # 找到最大突变的位置
    max_idx = action_diffs.argmax()
    max_diff = action_diffs[max_idx]

    print(f"\nEpisode {ep_idx}:")
    print(f"  长度: {length} 帧")
    print(f"  最大动作突变: {max_diff:.2f} 像素（阈值：50）")
    print(f"  突变位置: 帧 {max_idx} → 帧 {max_idx+1}（全局帧 {from_idx+max_idx} → {from_idx+max_idx+1}）")
    print(f"  动作变化统计:")
    print(f"    平均: {action_diffs.mean():.2f} 像素")
    print(f"    中位数: {np.median(action_diffs):.2f} 像素")
    print(f"    P95: {np.percentile(action_diffs, 95):.2f} 像素")
    print(f"    P99: {np.percentile(action_diffs, 99):.2f} 像素")

    # 显示突变前后的具体动作
    if max_idx < len(actions) - 1:
        action_before = actions[max_idx]
        action_after = actions[max_idx + 1]
        print(f"  突变详情:")
        print(f"    帧 {max_idx}: action = [{action_before[0]:.1f}, {action_before[1]:.1f}]")
        print(f"    帧 {max_idx+1}: action = [{action_after[0]:.1f}, {action_after[1]:.1f}]")
        print(f"    变化: Δx={action_after[0]-action_before[0]:.1f}, Δy={action_after[1]-action_before[1]:.1f}")

# 4. 统计所有动作突变的分布
print(f"\n" + "=" * 70)
print(f"📊 所有标记为异常的 Episode 的突变分布")
print("=" * 70)

all_max_jumps = []
for ep_detail in report['episode_details']:
    if 'max_action_jump' in ep_detail:
        all_max_jumps.append(ep_detail['max_action_jump'])

if len(all_max_jumps) > 0:
    all_max_jumps = np.array(all_max_jumps)
    print(f"\n动作突变统计（{len(all_max_jumps)} 个异常 Episode）:")
    print(f"  最小突变: {all_max_jumps.min():.2f} 像素")
    print(f"  最大突变: {all_max_jumps.max():.2f} 像素")
    print(f"  平均突变: {all_max_jumps.mean():.2f} 像素")
    print(f"  中位数: {np.median(all_max_jumps):.2f} 像素")

    # 分段统计
    range_50_60 = np.sum((all_max_jumps >= 50) & (all_max_jumps < 60))
    range_60_80 = np.sum((all_max_jumps >= 60) & (all_max_jumps < 80))
    range_80_100 = np.sum((all_max_jumps >= 80) & (all_max_jumps < 100))
    range_100_plus = np.sum(all_max_jumps >= 100)

    print(f"\n  突变范围分布:")
    print(f"    50-60 像素: {range_50_60} 个")
    print(f"    60-80 像素: {range_60_80} 个")
    print(f"    80-100 像素: {range_80_100} 个")
    print(f"    100+ 像素: {range_100_plus} 个")

# 5. 建议
print(f"\n" + "=" * 70)
print(f"💡 建议")
print("=" * 70)

if len(all_max_jumps) > 0:
    median_jump = np.median(all_max_jumps)

    if median_jump < 60:
        print(f"\n⚠️  大部分异常 Episode 的突变在 50-60 像素之间")
        print(f"   这接近阈值 50，可能是：")
        print(f"   1. 阈值设置得刚好在临界点，有点严格")
        print(f"   2. 数据确实有轻微的质量问题")
        print(f"\n   选项 A：放宽阈值到 60-70 像素")
        print(f"   选项 B：保持当前阈值，人工检查这些 Episode")
        print(f"   选项 C：查看被删除的 Episode 是否影响训练效果")

    if range_100_plus > 0:
        print(f"\n⚠️  有 {range_100_plus} 个 Episode 的突变 > 100 像素")
        print(f"   这些是明显的异常，应该删除")

print(f"\n✅ 分析完成！")
print("=" * 70)
