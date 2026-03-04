"""
Distributed Fine-Tuning on AKS with KubeRay
=============================================
Adapted from: https://docs.ray.io/en/latest/ray-overview/examples/entity-recognition-with-llms/README.html

This script performs:
  1. Data download and preparation
  2. Distributed fine-tuning with Ray Train + LLaMA-Factory

Usage (inside a RayJob, or via `ray job submit`):
  python fine_tune.py
"""

import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import ray
import yaml
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from IPython.display import Code, display

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "/tmp/viggo"
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

    # Fix file paths in dataset_info.json to point to the actual data_dir
    # (the S3 copy has hardcoded /mnt/cluster_storage/viggo/ paths)
    info_path = os.path.join(data_dir, "dataset_info.json")
    with open(info_path) as f:
        info = json.load(f)
    for ds in info.values():
        if "file_name" in ds:
            ds["file_name"] = os.path.join(data_dir, os.path.basename(ds["file_name"]))
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Data ready at {data_dir}")
    display(Code(filename=info_path, language="json"))
    return data_dir


def _distribute_files_to_gpu_nodes(file_data: dict, dest_dir: str, label: str) -> None:
    """
    Copy files to every alive GPU worker via Ray object store.
    Uses soft node affinity and retries for resilience against transient
    node failures (e.g. GCS heartbeat timeouts during pip install).
    """
    import time

    total_mb = sum(len(v) for v in file_data.values()) / 1024 / 1024
    print(f"Distributing {label} ({total_mb:.1f} MB, "
          f"{len(file_data)} files) to GPU workers …")

    data_ref = ray.put(file_data)

    @ray.remote(num_cpus=0.1, num_gpus=0)
    def _write_files(path: str, data):
        os.makedirs(path, exist_ok=True)
        for name, content in data.items():
            with open(os.path.join(path, name), "wb") as fh:
                fh.write(content)
        return True

    for attempt in range(5):
        nodes = [n for n in ray.nodes()
                 if n["Alive"] and n.get("Resources", {}).get("GPU", 0) > 0]
        if not nodes:
            print(f"  No alive GPU nodes found (attempt {attempt + 1}/5), waiting …")
            time.sleep(30)
            continue
        try:
            refs = []
            for node in nodes:
                strategy = NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=True)
                refs.append(
                    _write_files.options(scheduling_strategy=strategy)
                    .remote(dest_dir, data_ref)
                )
            ray.get(refs, timeout=600)
            print(f"  {label} distributed to {len(nodes)} GPU node(s).")
            return
        except Exception as e:
            print(f"  Distribution attempt {attempt + 1} failed: {e}")
            if attempt < 4:
                time.sleep(30)
    raise RuntimeError(f"Failed to distribute {label} to GPU workers after 5 attempts")


def distribute_data_to_workers(data_dir: str) -> None:
    """
    Copy dataset files from the head node to every GPU worker via Ray object
    store.  This removes the need for a shared PVC — each worker gets a local
    copy at the same path so LLaMA-Factory can read it normally.
    Also pre-creates the saves directory and Ray Train storage validation
    marker so that checkpoint persistence works with local storage.
    """
    file_data = {}
    for fname in DATA_FILES:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                file_data[fname] = f.read()
    _distribute_files_to_gpu_nodes(file_data, data_dir, "dataset")

    # Pre-create the saves directory and Ray Train validation marker on all
    # GPU nodes.  Ray Train checks that every node can read/write a marker
    # file at the storage_path; without shared storage this fails unless
    # the directory + marker already exist on every node.
    saves_dir = os.path.join(data_dir, "saves")
    marker_dir = os.path.join(saves_dir, "lora_sft_ray")
    os.makedirs(marker_dir, exist_ok=True)
    marker_path = os.path.join(marker_dir, ".validate_storage_marker")
    with open(marker_path, "w") as f:
        f.write("")  # empty marker
    _distribute_files_to_gpu_nodes(
        {".validate_storage_marker": b""}, marker_dir, "storage marker"
    )


# ============================================================
# 2. Distributed Fine-Tuning (Ray Train + LLaMA-Factory)
# ============================================================
def run_training(data_dir: str, num_workers: int = 4) -> str:
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
        # ray  — no Anyscale-specific resources
        "ray_run_name": "lora_sft_ray",
        "ray_storage_path": saves_dir,
        "ray_num_workers": num_workers,
        "resources_per_worker": {"GPU": 1},
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
    with open(config_path, "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    print("Training config written to", config_path)
    display(Code(filename=config_path, language="yaml"))

    # Launch training via LLaMA-Factory CLI with Ray backend
    env = {**os.environ, "USE_RAY": "1"}
    result = subprocess.run(
        ["llamafactory-cli", "train", config_path],
        env=env, capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"llamafactory-cli train failed with exit code {result.returncode}"
        )

    # Retrieve the LoRA checkpoint from GPU workers to the head node.
    # Ray Train saves checkpoints on the worker nodes, not the head.
    lora_path = _retrieve_checkpoint_from_workers(saves_dir)
    print(f"LoRA adapter saved at: {lora_path}")

    # Retrieve training results from GPU worker (written there by LLaMA-Factory)
    output_path = os.path.join(output_dir, "all_results.json")
    if not os.path.exists(output_path):
        _retrieve_file_from_workers(output_path)
    if os.path.exists(output_path):
        display(Code(filename=output_path, language="json"))
    return lora_path


def _retrieve_checkpoint_from_workers(saves_dir: str) -> str:
    """
    After training, the LoRA checkpoint lives on a GPU worker node.
    This function finds it, reads it via Ray object store, and writes
    it to the head node so the driver can use it for inference.
    """
    @ray.remote(num_cpus=0.1, num_gpus=0)
    def _find_and_read_checkpoint(saves_dir):
        base = Path(saves_dir) / "lora_sft_ray"
        if not base.exists():
            return None
        # Ray Tune creates TorchTrainer_<id> trial dirs with checkpoint_<n> inside
        trial_dirs = [d for d in base.iterdir()
                      if d.name.startswith("TorchTrainer_") and d.is_dir()]
        if not trial_dirs:
            # Fall back: look for checkpoint_* directly (Ray Train v2 layout)
            trial_dirs = [base]
        latest_trial = max(trial_dirs, key=lambda d: d.stat().st_mtime)
        ckpt_dirs = [d for d in latest_trial.iterdir()
                     if d.name.startswith("checkpoint_") and d.is_dir()]
        if not ckpt_dirs:
            return None
        latest_ckpt = max(ckpt_dirs, key=lambda d: d.stat().st_mtime)
        files = {}
        for f in latest_ckpt.rglob("*"):
            if f.is_file():
                files[str(f.relative_to(latest_ckpt))] = f.read_bytes()
        return {"rel_path": str(latest_ckpt.relative_to(Path(saves_dir))),
                "files": files}

    nodes = [n for n in ray.nodes()
             if n["Alive"] and n.get("Resources", {}).get("GPU", 0) > 0]
    for node in nodes:
        strategy = NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"], soft=True)
        result = ray.get(
            _find_and_read_checkpoint.options(scheduling_strategy=strategy)
            .remote(saves_dir), timeout=120
        )
        if result is not None:
            local_path = os.path.join(saves_dir, result["rel_path"])
            os.makedirs(local_path, exist_ok=True)
            for rel_name, content in result["files"].items():
                fpath = os.path.join(local_path, rel_name)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "wb") as f:
                    f.write(content)
            total_mb = sum(len(v) for v in result["files"].values()) / 1024 / 1024
            print(f"  Retrieved checkpoint ({total_mb:.1f} MB) from worker to head: {local_path}")
            return local_path

    raise FileNotFoundError("No checkpoint found on any GPU worker node")


def _retrieve_file_from_workers(file_path: str) -> None:
    """Retrieve a single file from a GPU worker node to the head."""
    @ray.remote(num_cpus=0.1, num_gpus=0)
    def _read_file(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    nodes = [n for n in ray.nodes()
             if n["Alive"] and n.get("Resources", {}).get("GPU", 0) > 0]
    for node in nodes:
        strategy = NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"], soft=True)
        content = ray.get(
            _read_file.options(scheduling_strategy=strategy)
            .remote(file_path), timeout=60)
        if content is not None:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            print(f"  Retrieved {file_path} from worker.")
            return
    print(f"  Warning: {file_path} not found on any GPU worker.")


# ============================================================
# Main
# ============================================================
def main():
    data_dir = DATA_DIR

    # ── Step 1: Data ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1  Downloading / verifying dataset")
    print("=" * 60)
    data_dir = download_data(data_dir)

    # Show a sample
    with open(os.path.join(data_dir, "train.jsonl")) as fp:
        sample = json.loads(fp.readline())
    print(f"\n  System prompt (first 120 chars):\n  {textwrap.shorten(sample['instruction'], 120)}")

    # Distribute dataset to GPU workers via Ray object store
    distribute_data_to_workers(data_dir)

    # ── Step 2: Training ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2  Distributed fine-tuning (Ray Train + LLaMA-Factory)")
    print("=" * 60)
    lora_path = run_training(data_dir)

    # ── Done ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All steps complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
