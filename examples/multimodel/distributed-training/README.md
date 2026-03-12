# Distributed Training on AKS with KubeRay

This example demonstrates **distributed model training** on Azure Kubernetes Service (AKS) using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Train](https://docs.ray.io/en/latest/train/train.html).

It is adapted from the [Ray E2E Multimodal AI Workloads — Distributed Training](https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/02-Distributed-Training.html) tutorial (originally designed for Anyscale) to run on AKS.

## What This Example Does

1. **Preprocess** — Uses [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) to embed dog breed images into 512-dimensional vectors using Ray Data `map_batches` with GPU actors.
2. **Train** — Trains a simple 2-layer PyTorch classifier with Ray Train `TorchTrainer` across multiple GPU workers using DDP.
3. **Track** — Logs metrics and model artifacts to an MLflow model registry on shared storage.
4. **Evaluate** — Runs batch inference on a held-out test set and computes precision, recall, F1, and accuracy.

## Prerequisites

| Component | Version / Details |
|---|---|
| AKS cluster | Created via `scripts/aks.sh` with GPU node pool |
| NVIDIA device plugin | Installed via `scripts/addon.sh` |
| KubeRay operator | v1.5.1, installed via `scripts/addon.sh` |
| Shared PVC | `cluster-storage` (100Gi ReadWriteMany), created via `configs/pvc.yaml` |
| Ray | 2.53.0 |

## Files

| File | Description |
|---|---|
| `distributed_training.py` | Training script — runs on the RayCluster |
| `rayjob.yaml` | RayJob manifest — submits the script and manages the cluster |
| `doggos/embed.py` | CLIP embedding actor for Ray Data |

## Architecture

```
kubectl apply -f rayjob.yaml
        │
        ▼
┌─────────────────────────────────────────────────┐
│ RayJob: distributed-training                    │
│                                                 │
│  Head Pod (CPU, system node pool)               │
│  ├── distributed_training.py (entrypoint)       │
│  ├── doggos/embed.py (CLIP actor)               │
│  └── /mnt/cluster_storage (Azure Blob PVC)      │
│                                                 │
│  Worker Pods (GPU) × 2                          │
│  ├── Ray Train workers (DDP)                    │
│  ├── Ray Data actors (CLIP embedding)           │
│  └── /mnt/cluster_storage (Azure Blob PVC)      │
└─────────────────────────────────────────────────┘
```

The training script is mounted via a ConfigMap. Pip dependencies (`torch`, `transformers`, `mlflow`, `doggos`, etc.) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`. The `doggos` helper package is installed directly from [GitHub](https://github.com/anyscale/multimodal-ai).

## Quick Start

### 1. Ensure PVC and StorageClass exist

```bash
kubectl apply -f configs/storageclass.yaml
kubectl apply -f configs/pvc.yaml
```

### 2. Create the ConfigMap

Package the training script so the RayJob pods can access it:

```bash
kubectl create configmap distributed-training-scripts \
    --from-file=distributed_training.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Submit the RayJob

```bash
kubectl apply -f rayjob.yaml
```

This creates a RayCluster (head + 2 GPU workers), installs pip dependencies via `runtimeEnvYAML`, runs `distributed_training.py`, and keeps the cluster alive for inspection.

### 4. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob distributed-training -w

# Stream logs
kubectl -n ray logs -f -l job-name=distributed-training --tail=100

# Ray Dashboard
kubectl -n ray port-forward svc/distributed-training-head-svc 8265:8265
```

### 5. View MLflow Metrics

```bash
# Run MLflow server inside the head pod
head_pod=$(kubectl get pod -l ray.io/node-type=head -o=jsonpath='{.items[0].metadata.name}'  -n ray)
kubectl -n ray exec -it $head_pod -- \
    mlflow server -h 0.0.0.0 -p 8080 \
    --backend-store-uri /mnt/cluster_storage/mlflow/doggos

# Port-forward in another terminal
kubectl -n ray port-forward $head_pod 8080:8080
```

MLflow UI is now available at [http://localhost:8080](http://localhost:8080).

## Adapting the Scaling Config

The default `rayjob.yaml` is configured for **2 GPU worker nodes with 8 GPUs each** (e.g., `Standard_ND96asr_v4`). Adjust for your setup:

| Node Pool VM SKU | GPUs/Node | `NUM_WORKERS` | `resources_per_worker` |
|---|---|---|---|
| `Standard_NC6s_v3` (V100) | 1 | 2 | `{"CPU": 4, "GPU": 1}` |
| `Standard_NC24ads_A100_v4` (A100) | 4 | 2 | `{"CPU": 8, "GPU": 4}` |
| `Standard_ND96asr_v4` (A100 x8) | 8 | 2 | `{"CPU": 8, "GPU": 8}` |

Update both `rayjob.yaml` (worker replicas and GPU limits) and the constants at the top of `distributed_training.py`.

## Key Differences from Anyscale Version

| Anyscale | AKS + KubeRay |
|---|---|
| Notebook runs inside Anyscale Workspace | Script runs on the cluster via RayJob |
| `accelerator_type="T4"` | Removed — GPU type determined by VM SKU |
| S3 user storage for artifacts | Azure Blob PVC at `/mnt/cluster_storage` |
| Anyscale runtime env auto-setup | `runtimeEnvYAML` in RayJob spec |
| `anyscale job submit` | `kubectl apply -f rayjob.yaml` |

## Cleanup

```bash
kubectl delete -f rayjob.yaml
kubectl delete configmap distributed-training-scripts -n ray
```
