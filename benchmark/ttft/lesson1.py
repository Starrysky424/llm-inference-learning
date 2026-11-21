
import json
import urllib.request
import time

url = "http://127.0.0.1:9931/v1/chat/completions"

data = {
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": True
}

# 将 Python 字典转换为 JSON 字节数据
body = json.dumps(data).encode("utf-8")

# 创建 HTTP POST 请求
request = urllib.request.Request(
    url,
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST"
)

# 记录请求开始时间
start = time.perf_counter()

# 记录第一个有效内容到达时间
first_content_time = None

# 发送请求
try:
    with urllib.request.urlopen(request) as response:

        # 逐行读取 SSE 流式响应
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()

            # 只处理 data: 开头的事件
            if not line.startswith("data:"):
                continue

            # 提取 data: 后面的内容
            json_text = line[5:].strip()

            # 判断流式响应是否结束
            if json_text == "[DONE]":
                break

            # 解析 JSON
            result = json.loads(json_text)

            # 获取本次新增的内容
            content = result["choices"][0]["delta"].get("content")

            if content:
                # 只记录第一次有效内容到达时间
                if first_content_time is None:
                    first_content_time = time.perf_counter()

                # 实时输出内容
                print(content, end="", flush=True)

    # 完整响应结束，记录结束时间
    end = time.perf_counter()

    print()

    # 计算耗时
    if first_content_time is not None:
        ttft = first_content_time - start
        total_time = end - start

        print(f"TTFT: {ttft:.4f} 秒")
        print(f"请求总耗时: {total_time:.4f} 秒")
    else:
        print("没有收到有效内容，TTFT 测量失败")

except Exception as e:
    print(f"请求发生错误: {e}")
