"""
Entity Recognition with LLMs on AKS with KubeRay
==================================================
Adapted from: https://docs.ray.io/en/latest/ray-overview/examples/entity-recognition-with-llms/README.html

This script performs:
  1. Data download and preparation
  2. Distributed fine-tuning with Ray Train + LLaMA-Factory
  3. Batch inference with ray.data.llm + vLLM
  4. Evaluation of results

Usage (inside a RayJob, or via `ray job submit`):
  python entity_recognition.py                     # full pipeline
  python entity_recognition.py --skip-training     # inference only (needs --lora-path)
  python entity_recognition.py --skip-inference    # training only
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import ray
import yaml
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from IPython.display import Code, display

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "/mnt/cluster_storage/viggo"
MODEL_SOURCE = "Qwen/Qwen2.5-7B-Instruct"
DATA_BASE_URL = "https://viggo-ds.s3.amazonaws.com"
DATA_FILES = ["train.jsonl", "val.jsonl", "test.jsonl", "dataset_info.json"]


# ============================================================
# 1. Data Ingestion
# ============================================================
def download_data(data_dir: str = DATA_DIR) -> str:
    """Download the viggo dataset."""
    os.makedirs(data_dir, exist_ok=True)

    for fname in DATA_FILES:
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(f"{DATA_BASE_URL}/{fname}", dest)
        else:
            print(f"{fname} already exists, skipping.")

    print(f"Data ready at {data_dir}")
    return data_dir


# ============================================================
# 2. Distributed Fine-Tuning (Ray Train + LLaMA-Factory)
# ============================================================
def run_training(
    data_dir: str, num_workers: int = 4, num_gpus_per_worker: int = 4
) -> str:
    """
    Run LoRA SFT using LLaMA-Factory with Ray Train.
    Returns the path to the saved LoRA adapter checkpoint.
    """
    output_dir = os.path.join(data_dir, "outputs")
    saves_dir = os.path.join(data_dir, "saves")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(saves_dir, exist_ok=True)

    training_config = {
        # model
        "model_name_or_path": MODEL_SOURCE,
        "trust_remote_code": True,
        # method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_target": "all",
        # dataset
        "dataset": "viggo-train",
        "dataset_dir": data_dir,
        "template": "qwen",
        "cutoff_len": 2048,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        # output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        # ray
        "ray_run_name": "lora_sft_ray",
        "ray_storage_path": saves_dir,
        "ray_num_workers": num_workers,
        "resources_per_worker": {"GPU": num_gpus_per_worker},
        "placement_strategy": "PACK",
        # train
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 5.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
        # eval
        "eval_dataset": "viggo-val",
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500,
    }

    config_path = os.path.join(data_dir, "lora_sft_ray.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    print("Training config written to", config_path)
    display(Code(filename=config_path, language="yaml"))

    # Launch training via LLaMA-Factory CLI with Ray backend, streaming output
    env = {**os.environ, "USE_RAY": "1"}
    process = subprocess.Popen(
        ["llamafactory-cli", "train", config_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"llamafactory-cli train failed with exit code {process.returncode}"
        )

    # Locate the latest LoRA checkpoint
    lora_path = _find_latest_checkpoint(saves_dir)
    print(f"LoRA adapter saved at: {lora_path}")
    output_path = os.path.join(output_dir, "all_results.json")
    display(Code(filename=output_path, language="json"))
    return lora_path


def _find_latest_checkpoint(saves_dir: str) -> str:
    save_dir = Path(saves_dir) / "lora_sft_ray"
    # Ray Train v2 (Ray ≥2.53) saves checkpoints as checkpoint_<timestamp> dirs
    checkpoint_dirs = [
        d for d in save_dir.iterdir() if d.name.startswith("checkpoint_") and d.is_dir()
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {save_dir}")
    latest = max(checkpoint_dirs, key=lambda d: d.stat().st_mtime)
    # The LoRA adapter files (adapter_model.safetensors, etc.) are
    # inside a "checkpoint" subfolder when persisted by Ray Train v2
    inner = latest / "checkpoint"
    return str(inner) if inner.is_dir() else str(latest)


# ============================================================
# 3. Batch Inference  (ray.data.llm + vLLM)
# ============================================================
def run_batch_inference(
    model_source: str, lora_path: str, data_dir: str, num_workers: int = 16
):
    """Run batch inference with vLLM through ray.data.llm and evaluate."""
    # System prompt from training data
    with open(os.path.join(data_dir, "train.jsonl"), "r", encoding="utf-8") as fp:
        system_content = json.loads(fp.readline())["instruction"]

    config = vLLMEngineProcessorConfig(
        model_source=model_source,
        runtime_env={
            "env_vars": {
                "VLLM_USE_V1": "0",  # v1 doesn't support LoRA adapters yet
            },
        },
        engine_kwargs={
            "enable_lora": True,
            "max_lora_rank": 8,
            "max_loras": 1,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
        },
        concurrency=num_workers,
        batch_size=16,
    )

    processor = build_processor(
        config,
        preprocess=lambda row: dict(
            model=lora_path,  # LoRA adapter path — remove this line for base-model-only inference
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["input"]},
            ],
            sampling_params={"temperature": 0.3, "max_tokens": 250},
        ),
        postprocess=lambda row: {**row, "generated_output": row["generated_text"]},
    )

    test_path = os.path.join(data_dir, "test.jsonl")
    ds = ray.data.read_json(test_path)
    ds = processor(ds)
    results = ds.take_all()

    print("\nSample inference result:")
    display(Code(json.dumps(results[0], indent=2, default=str), language="json"))

    # --- Evaluation (exact match) ---
    matches = sum(1 for r in results if r["output"] == r["generated_output"])
    accuracy = matches / len(results) if results else 0.0
    return results, matches, accuracy


# ============================================================
# Main
# ============================================================
def main():
    """Main function to run the full pipeline."""

    parser = argparse.ArgumentParser(
        description="E2E LLM Entity Recognition on AKS with KubeRay"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; requires --lora-path or an existing checkpoint",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip batch inference (training only)",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help="Dataset directory (default: %(default)s)"
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Path to an existing LoRA adapter (skips training)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Ray Train GPU workers (default: 4)",
    )
    args = parser.parse_args()

    # ── Step 1: Data Ingestion ──────────────────────────────────
    print("=" * 60)
    print("Step 1  Downloading / verifying dataset")
    print("=" * 60)
    data_dir = download_data(args.data_dir)

    # Show a sample
    with open(os.path.join(data_dir, "train.jsonl"), "r", encoding="utf-8") as fp:
        sample = json.loads(fp.readline())
    print("\nSample training instance:")
    display(Code(json.dumps(sample, indent=2, ensure_ascii=False), language="json"))

    # Show dataset info
    print("Dataset info:")
    display(Code(filename=os.path.join(data_dir, "dataset_info.json"), language="json"))

    # ── Step 2: Training ────────────────────────────────────────
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("Step 2  Distributed fine-tuning (Ray Train + LLaMA-Factory)")
        print("=" * 60)
        lora_path = run_training(data_dir, num_workers=args.num_workers)
    else:
        lora_path = args.lora_path
        if not lora_path:
            try:
                lora_path = _find_latest_checkpoint(os.path.join(data_dir, "saves"))
            except FileNotFoundError:
                print(
                    "ERROR: --lora-path not provided and no existing checkpoint found."
                )
                sys.exit(1)
        print(f"Using existing LoRA adapter: {lora_path}")

    # ── Step 3: Batch Inference ─────────────────────────────────
    if not args.skip_inference:
        print("\n" + "=" * 60)
        print("Step 3  Batch inference (ray.data.llm + vLLM)")
        print("=" * 60)

        results, matches, accuracy = run_batch_inference(
            MODEL_SOURCE, lora_path, data_dir
        )

        print("Batch Inference Results")
        print("─────────────────────────")
        print(f"Total samples : {len(results)}")
        print(f"Exact matches : {matches}")
        print(f"Accuracy      : {accuracy:.4f}")

    # ── Done ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All steps complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
