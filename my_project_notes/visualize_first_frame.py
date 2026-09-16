#!/usr/bin/env python3
"""
可视化 LeRobot 数据集的第一帧图像

学习目标：
1. 理解 PyTorch Tensor 图像格式
2. 学会 [C, H, W] 到 [H, W, C] 的转换
3. 保存和查看图像
"""

import torch
from lerobot.datasets import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("LeRobot 数据可视化脚本")
print("=" * 60)

# 1. 加载数据集
print("\n[1/5] 加载数据集...")
dataset = LeRobotDataset('lerobot/pusht')
print(f"✅ 数据集加载成功：{dataset.repo_id}")
print(f"   总帧数: {len(dataset)}")
print(f"   Episode 数量: {dataset.num_episodes}")

# 2. 获取第 0 帧数据
print("\n[2/5] 获取第 0 帧数据...")
frame_0 = dataset[0]
print(f"✅ 第 0 帧包含 {len(frame_0)} 个字段")

# 3. 提取图像 Tensor
print("\n[3/5] 提取图像数据...")
image_tensor = frame_0['observation.image']
print(f"✅ 图像 shape: {image_tensor.shape}")
print(f"   数据类型: {image_tensor.dtype}")
print(f"   数值范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

# 4. 转换格式用于显示
print("\n[4/5] 转换图像格式...")
print(f"   原始格式 (PyTorch): {image_tensor.shape} = [C, H, W]")

# PyTorch: [C, H, W] → Matplotlib: [H, W, C]
image_numpy = image_tensor.permute(1, 2, 0).numpy()
print(f"   转换后格式: {image_numpy.shape} = [H, W, C]")

# 检查数值范围（matplotlib 需要 [0, 1] 或 [0, 255]）
if image_numpy.max() <= 1.0:
    print(f"   ✅ 数值范围 [0, 1]，可以直接显示")
    display_image = image_numpy
else:
    print(f"   ⚠️  数值范围 [{image_numpy.min()}, {image_numpy.max()}]")
    print(f"   正在归一化到 [0, 1]...")
    display_image = np.clip(image_numpy / 255.0, 0, 1)

# 5. 保存图像
print("\n[5/5] 保存图像...")
output_path = "/home/kinala/lerebot/lerobot/my_project_notes/frame_0_observation.png"

plt.figure(figsize=(8, 8))
plt.imshow(display_image)
plt.title(f"Episode 0, Frame 0\nTask: {frame_0['task']}", fontsize=10)
plt.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 图像已保存到: {output_path}")

# 额外信息：显示这一帧的其他数据
print("\n" + "=" * 60)
print("第 0 帧的其他数据：")
print("=" * 60)
print(f"机器人状态 (observation.state): {frame_0['observation.state'].numpy()}")
print(f"执行的动作 (action): {frame_0['action'].numpy()}")
print(f"时间戳 (timestamp): {frame_0['timestamp'].item():.3f} 秒")
print(f"Episode 索引: {frame_0['episode_index'].item()}")
print(f"Frame 索引: {frame_0['frame_index'].item()}")
print(f"任务: {frame_0['task']}")

print("\n" + "=" * 60)
print("✅ 可视化完成！")
print("=" * 60)
print(f"\n查看图像：")
print(f"  在文件管理器中打开: {output_path}")
print(f"  或使用命令: xdg-open {output_path}")
