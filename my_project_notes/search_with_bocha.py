#!/usr/bin/env python3
"""
使用 Bocha API 搜索机器人数据质量规则

用途：调用 Bocha API 搜索关于如何定义机器人数据质量规则的信息
"""

import os
import requests
import json

# 1. 检查环境变量
api_key = os.environ.get('BOCHA_API_KEY')
if not api_key:
    print("❌ 错误：未找到 BOCHA_API_KEY 环境变量")
    print("请设置：export BOCHA_API_KEY='your-key'")
    exit(1)

print("✅ 找到 BOCHA_API_KEY")

# 2. 定义搜索问题
query = """
我正在做机器人数据质量检查项目，使用 LeRobot 的 pusht 数据集。

请详细解释：

1. 如何为不同的机器人任务定义数据质量规则？
   - 通用规则 vs 任务特定规则
   - 如何确定阈值（例如 Episode 过短、动作突变）

2. 不同类型的机器人任务（如 pusht、aloha、pick-and-place）的数据质量规则有什么区别？

3. 在实际工程中，如何从数据统计中发现异常并定义规则？

4. 有哪些常见的机器人数据质量问题和检测方法？

请给出具体的例子和数值范围。
"""

# 3. 调用 Bocha API（OpenAI 兼容格式）
url = "https://api.bocha.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "deepseek-v4-pro",
    "messages": [
        {
            "role": "system",
            "content": "你是一个机器人数据工程和机器学习数据质量方面的专家。请提供详细、具体、可操作的建议。"
        },
        {
            "role": "user",
            "content": query
        }
    ],
    "stream": False,
    "temperature": 0.7,
    "max_tokens": 4096
}

print("\n" + "=" * 70)
print("正在调用 Bocha API 搜索...")
print("=" * 70)

try:
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()

    # 4. 提取回答
    if 'choices' in result and len(result['choices']) > 0:
        answer = result['choices'][0]['message']['content']

        print("\n" + "=" * 70)
        print("Bocha API 搜索结果")
        print("=" * 70)
        print(answer)

        # 5. 保存到文件
        output_path = "/home/kinala/lerebot/lerobot/my_project_notes/bocha_search_result.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Bocha API 搜索结果：机器人数据质量规则\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"查询问题：\n{query}\n\n")
            f.write("=" * 70 + "\n")
            f.write("回答：\n")
            f.write("=" * 70 + "\n\n")
            f.write(answer)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("使用的模型：deepseek-v4-pro\n")
            f.write("=" * 70 + "\n")

        print("\n" + "=" * 70)
        print(f"✅ 搜索结果已保存到：{output_path}")
        print("=" * 70)

    else:
        print("❌ 错误：API 返回格式异常")
        print(json.dumps(result, indent=2, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"❌ API 调用失败：{e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"状态码：{e.response.status_code}")
        print(f"响应内容：{e.response.text}")
except Exception as e:
    print(f"❌ 发生错误：{e}")
