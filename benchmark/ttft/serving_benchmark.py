
import json
import time
import csv
import statistics
import urllib.request
import urllib.error


# ==============================
# 1. 基础配置
# ==============================

SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"

# 固定 Prompt，保证不同实验组的输入一致
PROMPT = "请用简单的语言解释什么是 C++，并举例说明。"

# 每组测试的最大生成 Token 数
OUTPUT_LENGTHS = [64, 128, 256]

# 每组重复测试次数
REPEAT_TIMES = 5

# CSV 结果文件
RESULT_FILE = "serving_benchmark_results.csv"


# ==============================
# 2. 计算统计指标
# ==============================

def calculate_latency(start_time, end_time):
    """
    计算两个时间点之间的耗时，单位为秒。
    """
    return end_time - start_time


def calculate_statistics(values):
    """
    计算平均值、标准差、最小值和最大值。
    """
    return {
        "average": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
        "minimum": min(values),
        "maximum": max(values)
    }


# ==============================
# 3. 发送一次流式请求
# ==============================

def run_single_request(max_tokens):

    # 构造请求数据
    request_data = {
    "model": "ggml-org/Qwen3.5-0.8B-GGUF:Q8_0",
    "messages": [
        {
            "role": "user",
            "content": PROMPT
        }
    ],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
    "reasoning_effort": "none",
    "stream_options": {
        "include_usage": True
    }
}

    # Python 字典转换为 JSON 字节
    json_data = json.dumps(request_data).encode("utf-8")

    # 创建 HTTP POST 请求
    request = urllib.request.Request(
        SERVER_URL,
        data=json_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    # 记录请求开始时间
    start_time = time.perf_counter()

    # 初始化变量
    first_token_time = None
    output_text = ""
    completion_tokens = None

    try:
        # 发送请求并读取 SSE 流
        with urllib.request.urlopen(request, timeout=300) as response:

            while True:

                line = response.readline()

                # 没有更多数据，结束读取
                if not line:
                    break

                line = line.decode("utf-8").strip()

                # 只处理 SSE 的 data 行
                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                # SSE 结束标记
                if data == "[DONE]":
                    break

                # 解析 JSON
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # 获取 usage 中的 Token 数
                usage = chunk.get("usage")

                if usage is not None:
                    completion_tokens = usage.get("completion_tokens")

                # 获取本次流式响应的 choices
                choices = chunk.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                # 只处理非空文本
                if content:
                    # 第一次收到有效内容时记录时间
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    output_text += content

        # 完整响应结束时间
        end_time = time.perf_counter()

    except urllib.error.HTTPError as error:
        print("HTTP 请求失败:", error.code)
        print(error.read().decode("utf-8", errors="replace"))
        raise

    except urllib.error.URLError as error:
        print("无法连接 llama-server:", error.reason)
        raise

    # 计算 TTFT
    if first_token_time is None:
        raise RuntimeError("没有收到有效输出内容，无法计算 TTFT")

    ttft = calculate_latency(start_time, first_token_time)

    # 计算端到端延迟
    e2e_latency = calculate_latency(start_time, end_time)

    # 生成阶段耗时
    generation_time = e2e_latency - ttft

    # 初始化指标
    tpot = None
    tokens_per_second = None

    # 有真实 Token 数时才计算生成速度
    if completion_tokens is not None and completion_tokens > 1:
        tpot = generation_time / (completion_tokens - 1)
        tokens_per_second = (completion_tokens - 1) / generation_time

    return {
        "ttft": ttft,
        "e2e_latency": e2e_latency,
        "generation_time": generation_time,
        "completion_tokens": completion_tokens,
        "tpot": tpot,
        "tokens_per_second": tokens_per_second,
        "output_chars": len(output_text)
    }


# ==============================
# 4. 执行完整 Benchmark
# ==============================

def run_benchmark():

    all_results = []

    for max_tokens in OUTPUT_LENGTHS:

        print()
        print("=" * 50)
        print("当前测试生成长度:", max_tokens)
        print("=" * 50)

        for run_index in range(REPEAT_TIMES):

            result = run_single_request(max_tokens)

            result["max_tokens"] = max_tokens
            result["run"] = run_index + 1

            all_results.append(result)

            print(
                f"第 {run_index + 1} 次 | "
                f"TTFT: {result['ttft']:.4f}s | "
                f"E2E: {result['e2e_latency']:.4f}s | "
                f"Tokens: {result['completion_tokens']} | "
                f"字符数: {result['output_chars']}"
            )

    return all_results


# ==============================
# 5. 保存 CSV 文件
# ==============================

def save_results(results):

    fieldnames = [
        "max_tokens",
        "run",
        "ttft",
        "e2e_latency",
        "generation_time",
        "completion_tokens",
        "tpot",
        "tokens_per_second",
        "output_chars"
    ]

    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(results)

    print()
    print("测试结果已保存:", RESULT_FILE)


# ==============================
# 6. 计算每组统计结果
# ==============================

def print_summary(results):

    print()
    print("#" * 65)
    print("Serving Benchmark 汇总")
    print("#" * 65)

    for max_tokens in OUTPUT_LENGTHS:

        group = [
            result for result in results
            if result["max_tokens"] == max_tokens
        ]

        ttft_values = [item["ttft"] for item in group]
        e2e_values = [item["e2e_latency"] for item in group]

        ttft_stats = calculate_statistics(ttft_values)
        e2e_stats = calculate_statistics(e2e_values)

        print()
        print(f"生成长度: {max_tokens}")

        print(
            f"TTFT 平均值: {ttft_stats['average']:.4f}s | "
            f"标准差: {ttft_stats['stddev']:.4f}s"
        )

        print(
            f"E2E 平均值: {e2e_stats['average']:.4f}s | "
            f"标准差: {e2e_stats['stddev']:.4f}s"
        )

        print(
            f"TTFT 范围: "
            f"{ttft_stats['minimum']:.4f}s ~ "
            f"{ttft_stats['maximum']:.4f}s"
        )

        print(
            f"E2E 范围: "
            f"{e2e_stats['minimum']:.4f}s ~ "
            f"{e2e_stats['maximum']:.4f}s"
        )

        # 统计有真实 Token 数的请求
        valid_results = [
            item for item in group
            if item["tpot"] is not None
        ]

        if valid_results:
            tpot_values = [item["tpot"] for item in valid_results]
            speed_values = [
                item["tokens_per_second"] for item in valid_results
            ]

            print(
                f"TPOT 平均值: "
                f"{statistics.mean(tpot_values):.6f}s/token"
            )

            print(
                f"生成速度平均值: "
                f"{statistics.mean(speed_values):.2f} tokens/s"
            )
        else:
            print("没有收到有效 Token usage，暂不计算 TPOT 和 Tokens/s")


# ==============================
# 7. 程序入口
# ==============================

if __name__ == "__main__":

    results = run_benchmark()

    save_results(results)

    print_summary(results)

    print()
    print("Benchmark 测试完成！")
