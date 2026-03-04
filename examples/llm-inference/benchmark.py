"""
Simple LLM Inference Benchmark
===============================
Measures throughput and latency of LLM inference using vLLM.
No Ray or distributed framework required — runs on a single node.

Usage:
  python benchmark.py
  python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --num-prompts 100
  python benchmark.py --tensor-parallel-size 4
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams


PROMPT_TEMPLATES = [
    "Explain the concept of gravitational waves in simple terms.",
    "Write a Python function that implements binary search.",
    "What are the key differences between TCP and UDP?",
    "Summarize the process of photosynthesis step by step.",
    "Describe the architecture of a modern CPU.",
    "What is the CAP theorem in distributed systems?",
    "Explain how transformers work in machine learning.",
    "Write a SQL query to find the second highest salary.",
    "What causes the northern lights?",
    "Describe the difference between concurrency and parallelism.",
]


def build_prompts(num_prompts: int) -> list[str]:
    """Repeat prompt templates to reach the desired count."""
    prompts = []
    for i in range(num_prompts):
        prompts.append(PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)])
    return prompts


def run_benchmark(
    model: str,
    num_prompts: int,
    max_tokens: int,
    tensor_parallel_size: int,
) -> dict:
    """Run inference benchmark and return metrics."""

    print(f"Loading model: {model}")
    print(f"  tensor_parallel_size = {tensor_parallel_size}")
    load_start = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time:.1f}s")

    prompts = build_prompts(num_prompts)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)

    print(f"\nRunning inference on {num_prompts} prompts (max_tokens={max_tokens}) ...")
    gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.perf_counter() - gen_start

    # Collect metrics
    total_input_tokens = 0
    total_output_tokens = 0
    for output in outputs:
        total_input_tokens += len(output.prompt_token_ids)
        total_output_tokens += sum(len(o.token_ids) for o in output.outputs)

    tokens_per_sec = total_output_tokens / gen_time if gen_time > 0 else 0
    avg_latency = gen_time / num_prompts

    metrics = {
        "model": model,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "model_load_time_s": round(load_time, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "generation_time_s": round(gen_time, 2),
        "output_tokens_per_sec": round(tokens_per_sec, 2),
        "avg_latency_per_request_s": round(avg_latency, 3),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simple LLM inference benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (default: %(default)s)")
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of prompts to benchmark (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens per prompt (default: %(default)s)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: %(default)s)")
    args = parser.parse_args()

    metrics = run_benchmark(
        model=args.model,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print("\n" + "=" * 50)
    print("  Benchmark Results")
    print("=" * 50)
    for key, val in metrics.items():
        print(f"  {key:30s} : {val}")
    print("=" * 50)

    # Also dump as JSON for easy parsing
    print(f"\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
