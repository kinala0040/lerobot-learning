#!/usr/bin/env python3
"""
视频解码实践：提取帧并测试性能

目标：
  1. 从 MP4 视频中提取单帧
  2. 测试解码速度
  3. 可视化提取的帧
  4. 对比缓存命中率
"""

import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

print("=" * 70)
print("LeRobot 视频解码探索")
print("=" * 70)

# 1. 找到视频文件
print("\n[步骤 1/5] 查找视频文件...")

video_path = None
cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "hub"

for item in cache_dir.glob("*pusht*"):
    if item.is_dir():
        # 查找视频文件
        video_files = list(item.rglob("*.mp4"))
        if video_files:
            video_path = video_files[0]
            print(f"✅ 找到视频: {video_path}")
            print(f"   大小: {video_path.stat().st_size / (1024*1024):.2f} MB")
            break

if not video_path:
    print("❌ 未找到视频文件")
    exit(1)

# 2. 使用 PyAV 解码视频
print("\n[步骤 2/5] 使用 PyAV 解码视频...")

try:
    import av

    # 打开视频
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    print(f"✅ 视频信息:")
    print(f"   编码格式: {video_stream.codec_context.name}")
    print(f"   分辨率: {video_stream.width}×{video_stream.height}")
    print(f"   总帧数: {video_stream.frames}")
    print(f"   FPS: {video_stream.average_rate}")
    print(f"   时长: {float(video_stream.duration * video_stream.time_base):.2f} 秒")

except ImportError:
    print("❌ PyAV 未安装")
    print("   安装命令: uv pip install av")
    exit(1)

# 3. 提取单帧
print("\n[步骤 3/5] 提取视频帧...")

# 要提取的帧索引
frame_indices = [0, 100, 1000, 5000, 10000, 20000]

output_dir = Path("my_project_notes/video_frames")
output_dir.mkdir(exist_ok=True)

for target_idx in frame_indices:
    if target_idx >= video_stream.frames:
        continue

    # 重新打开视频（PyAV 的 seek 不太稳定）
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    # 定位到目标帧
    start_time = time.time()

    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx == target_idx:
            # 转换为 NumPy 数组
            img = frame.to_ndarray(format='rgb24')

            # 保存为图像
            output_path = output_dir / f"frame_{target_idx:05d}.png"
            Image.fromarray(img).save(output_path)

            decode_time = time.time() - start_time
            print(f"   帧 {target_idx:5d}: {img.shape}, 耗时 {decode_time:.4f}s, 保存到 {output_path.name}")
            break

    container.close()

print(f"\n✅ 已保存到: {output_dir}/")

# 4. 性能测试：顺序访问 vs 随机访问
print("\n[步骤 4/5] 性能测试...")

# 测试 1: 顺序访问（缓存友好）
print("\n📊 测试 1: 顺序访问 100 帧")
container = av.open(str(video_path))
frames_sequential = []
start = time.time()

for frame_idx, frame in enumerate(container.decode(video=0)):
    if frame_idx >= 100:
        break
    frames_sequential.append(frame.to_ndarray(format='rgb24'))

time_sequential = time.time() - start
container.close()

print(f"   总耗时: {time_sequential:.4f} 秒")
print(f"   平均每帧: {time_sequential / 100 * 1000:.2f} 毫秒")

# 测试 2: 随机访问（缓存不友好）
print("\n📊 测试 2: 随机访问 10 帧")
random_indices = [50, 500, 1500, 3000, 5000, 8000, 12000, 15000, 18000, 22000]
frames_random = []
start = time.time()

for target_idx in random_indices:
    if target_idx >= video_stream.frames:
        continue

    container = av.open(str(video_path))

    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx == target_idx:
            frames_random.append(frame.to_ndarray(format='rgb24'))
            break

    container.close()

time_random = time.time() - start

print(f"   总耗时: {time_random:.4f} 秒")
print(f"   平均每帧: {time_random / len(frames_random) * 1000:.2f} 毫秒")

# 5. 对比分析
print("\n[步骤 5/5] 性能对比...")

print(f"\n📊 性能对比:")
print(f"   顺序访问: {time_sequential / 100 * 1000:.2f} ms/帧")
print(f"   随机访问: {time_random / len(frames_random) * 1000:.2f} ms/帧")
print(f"   性能差异: {(time_random / len(frames_random)) / (time_sequential / 100):.2f}x")

print("\n" + "=" * 70)
print("✅ 探索完成！")
print("=" * 70)

print("\n💡 关键发现:")
print("   1. 视频解码需要时间（每帧 5-20 ms）")
print("   2. 顺序访问比随机访问快（利用解码缓存）")
print("   3. 训练时应该尽量顺序读取连续的帧")
print("   4. 多进程并行解码可以提高吞吐量")

print(f"\n📁 提取的帧保存在: {output_dir}/")
print("   可以用图像查看器打开查看")
