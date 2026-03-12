"""
LLM Inference Benchmark on RayCluster
======================================
Measures throughput and latency of LLM inference using ray.data.llm
(backed by vLLM) on a RayCluster managed by KubeRay on AKS.

The script builds a Ray Data pipeline that preprocesses prompts into
chat messages, runs them through the vLLM engine, and collects
throughput / latency metrics from the results.

Usage (local / standalone):
  python benchmark.py
  python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --num-prompts 100
  python benchmark.py --tensor-parallel-size 4
"""

import argparse
import json
import os
import time

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_processor


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


def build_prompts(num_prompts: int) -> list[dict]:
    """Build prompt dicts for a Ray Dataset."""
    return [
        {"prompt": PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]}
        for i in range(num_prompts)
    ]


def run_benchmark(
    model: str,
    num_prompts: int,
    max_tokens: int,
    tensor_parallel_size: int,
    concurrency: int,
) -> dict:
    """Run inference benchmark and return metrics."""

    total_gpus = tensor_parallel_size * concurrency
    # ── 1. Build vLLM processor via ray.data.llm ────────────────────────
    print(f"Configuring vLLM processor: {model}")
    print(f"  tensor_parallel_size = {tensor_parallel_size}")
    print(f"  concurrency (engine replicas) = {concurrency}")
    print(f"  total GPUs = {total_gpus}")

    config = vLLMEngineProcessorConfig(
        model_source=model,
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
        },
        concurrency=concurrency,
        batch_size=64,
    )

    processor = build_processor(
        config,
        preprocess=lambda row: {
            "messages": [{"role": "user", "content": row["prompt"]}],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            },
        },
        postprocess=lambda row: {
            "prompt": row["prompt"],
            "generated_text": row["generated_text"],
            "num_input_tokens": row["num_input_tokens"],
            "num_output_tokens": row["num_generated_tokens"],
        },
    )

    # ── 2. Create dataset and run inference ─────────────────────────────
    ds = ray.data.from_items(build_prompts(num_prompts))

    print(f"\nRunning inference on {num_prompts} prompts (max_tokens={max_tokens}) ...")
    gen_start = time.perf_counter()
    result_ds = processor(ds)
    results = result_ds.take_all()
    gen_time = time.perf_counter() - gen_start

    # ── 3. Aggregate metrics ────────────────────────────────────────────
    total_input_tokens = sum(r["num_input_tokens"] for r in results)
    total_output_tokens = sum(r["num_output_tokens"] for r in results)

    tokens_per_sec = total_output_tokens / gen_time if gen_time > 0 else 0
    avg_latency = gen_time / num_prompts

    metrics = {
        "model": model,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "concurrency": concurrency,
        "total_gpus": tensor_parallel_size * concurrency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "generation_time_s": round(gen_time, 2),
        "output_tokens_per_sec": round(tokens_per_sec, 2),
        "avg_latency_per_request_s": round(avg_latency, 3),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="LLM inference benchmark on RayCluster (ray.data.llm)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=int(os.environ.get("NUM_PROMPTS", "50")),
        help="Number of prompts to benchmark (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", "256")),
        help="Max output tokens per prompt (default: 256)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
        help="Number of GPUs for tensor parallelism per engine (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "1")),
        help="Number of parallel engine replicas (default: 1)",
    )
    args = parser.parse_args()

    # Initialize Ray — connects to the existing RayCluster when submitted
    # via RayJob, or starts a local cluster when run standalone.
    ray.init()

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    metrics = run_benchmark(
        model=args.model,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        concurrency=args.concurrency,
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
