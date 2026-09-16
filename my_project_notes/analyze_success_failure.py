#!/usr/bin/env python3
"""
检查 pusht 数据集的成功/失败情况

目标：理解数据集中成功和失败的 Episode 分布
"""

from lerobot.datasets import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Episode 成功/失败分析")
print("=" * 70)

# 加载数据集
dataset = LeRobotDataset('lerobot/pusht')

print(f"\n数据集: {dataset.repo_id}")
print(f"总 Episode 数量: {dataset.num_episodes}")

# 收集每个 Episode 的信息
episode_info = []

for ep_idx in range(dataset.num_episodes):
    from_idx = dataset.meta.episodes['dataset_from_index'][ep_idx]
    to_idx = dataset.meta.episodes['dataset_to_index'][ep_idx]
    length = to_idx - from_idx

    # 获取最后一帧
    last_frame = dataset[to_idx - 1]

    # 检查是否成功
    success = last_frame['next.success'].item() if 'next.success' in last_frame else False
    done = last_frame['next.done'].item() if 'next.done' in last_frame else True
    reward = last_frame['next.reward'].item() if 'next.reward' in last_frame else 0.0

    episode_info.append({
        'episode': ep_idx,
        'length': length,
        'duration': length / dataset.fps,
        'success': success,
        'done': done,
        'reward': reward
    })

# 统计
total = len(episode_info)
success_count = sum(1 for ep in episode_info if ep['success'])
failure_count = total - success_count
success_rate = success_count / total * 100

print(f"\n" + "=" * 70)
print("📊 成功/失败统计")
print("=" * 70)
print(f"总 Episode 数: {total}")
print(f"成功: {success_count} ({success_rate:.1f}%)")
print(f"失败: {failure_count} ({100 - success_rate:.1f}%)")

# 成功和失败的长度统计
success_lengths = [ep['length'] for ep in episode_info if ep['success']]
failure_lengths = [ep['length'] for ep in episode_info if not ep['success']]

if len(success_lengths) > 0:
    print(f"\n成功 Episode 的长度:")
    print(f"  平均: {np.mean(success_lengths):.1f} 帧 ({np.mean(success_lengths)/dataset.fps:.1f} 秒)")
    print(f"  最短: {np.min(success_lengths)} 帧 ({np.min(success_lengths)/dataset.fps:.1f} 秒)")
    print(f"  最长: {np.max(success_lengths)} 帧 ({np.max(success_lengths)/dataset.fps:.1f} 秒)")

if len(failure_lengths) > 0:
    print(f"\n失败 Episode 的长度:")
    print(f"  平均: {np.mean(failure_lengths):.1f} 帧 ({np.mean(failure_lengths)/dataset.fps:.1f} 秒)")
    print(f"  最短: {np.min(failure_lengths)} 帧 ({np.min(failure_lengths)/dataset.fps:.1f} 秒)")
    print(f"  最长: {np.max(failure_lengths)} 帧 ({np.max(failure_lengths)/dataset.fps:.1f} 秒)")

# 查看一些具体的失败案例
if failure_count > 0:
    print(f"\n" + "=" * 70)
    print("🔍 失败案例（前 10 个）")
    print("=" * 70)
    failure_episodes = [ep for ep in episode_info if not ep['success']][:10]
    for ep in failure_episodes:
        print(f"Episode {ep['episode']:3d}: {ep['length']:3d} 帧 ({ep['duration']:5.1f}s), "
              f"reward={ep['reward']:.3f}, done={ep['done']}")

# 可视化
if success_count > 0 and failure_count > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: 成功率饼图
    axes[0].pie([success_count, failure_count],
                labels=['Success', 'Failure'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%',
                startangle=90)
    axes[0].set_title(f'Success Rate: {success_rate:.1f}%')

    # 图2: 成功和失败的长度分布
    axes[1].hist([success_lengths, failure_lengths],
                 bins=20,
                 label=['Success', 'Failure'],
                 color=['lightgreen', 'lightcoral'],
                 alpha=0.7,
                 edgecolor='black')
    axes[1].set_xlabel('Episode Length (frames)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Episode Length: Success vs Failure')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "/home/kinala/lerebot/lerobot/my_project_notes/success_failure_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {output_path}")

print("\n" + "=" * 70)
print("✅ 分析完成")
print("=" * 70)

# 关键观察
print("\n💡 关键观察:")
if success_rate == 0:
    print("  ⚠️  所有 Episode 都标记为失败（success=False）")
    print("  这可能是因为：")
    print("    1. 数据集收集时没有设置 success 标志")
    print("    2. 数据来自人类演示的'过程'，而非'成功案例'")
    print("    3. 模仿学习不需要 success 标志，只需要人类动作")
elif success_rate == 100:
    print("  ✅ 所有 Episode 都是成功案例")
    print("  这是理想的模仿学习数据集！")
else:
    print(f"  📊 数据集包含 {success_rate:.1f}% 成功和 {100-success_rate:.1f}% 失败案例")
    print("  混合数据集可以用于：")
    print("    - 训练模型识别成功和失败")
    print("    - 对比学习（contrastive learning）")
    print("    - 但模仿学习通常只用成功案例")
