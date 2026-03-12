# Distributed Training on AKS with KubeRay

This example demonstrates **distributed model training** on Kubernetes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Train](https://docs.ray.io/en/latest/train/train.html). It supports both **Azure (AKS)** and **Nebius** clusters.

It is adapted from the [Ray E2E Multimodal AI Workloads — Distributed Training](https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/02-Distributed-Training.html) tutorial (originally designed for Anyscale).

## What This Example Does

1. **Preprocess** — Uses [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) to embed dog breed images into 512-dimensional vectors using Ray Data `map_batches` with GPU actors.
2. **Train** — Trains a simple 2-layer PyTorch classifier with Ray Train `TorchTrainer` across multiple GPU workers using DDP.
3. **Track** — Logs metrics and model artifacts to an MLflow model registry on shared storage.
4. **Evaluate** — Runs batch inference on a held-out test set and computes precision, recall, F1, and accuracy.

## Prerequisites

| Component | Version / Details |
|---|---|
| Kubernetes cluster | AKS or Nebius with GPU node pool |
| NVIDIA GPU DRA driver | `gpu.nvidia.com` device class available on GPU nodes |
| KubeRay operator | v1.5.1 (installed by `run.sh` if not present) |
| Shared PVC | `cluster-storage` (100Gi ReadWriteMany), created via `configs/pvc.yaml` |
| Ray | 2.48.0 |

## Directory Structure

```
distributed-training/
├── main.py                          # Training script (runs on the RayCluster)
├── run.sh                           # One-command launcher (azure or nebius)
├── doggos/embed.py                  # CLIP embedding actor for Ray Data
├── base/
│   ├── kustomization.yaml           # Kustomize base
│   ├── rayjob.yaml                  # Cloud-agnostic RayJob manifest
│   └── gpu-claim.yaml               # DRA ResourceClaimTemplate (8 GPUs per worker)
└── overlays/
    ├── azure/
    │   ├── kustomization.yaml       # Azure overlay
    │   └── rayjob-patch.yaml        # nodeSelector for Azure
    └── nebius/
        ├── kustomization.yaml       # Nebius overlay
        └── rayjob-patch.yaml        # nodeSelector for Nebius
```

GPU allocation is defined in `base/gpu-claim.yaml` as a standalone [ResourceClaimTemplate](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) (`multi-gpu`) that requests 8 NVIDIA H100 GPUs per worker via the `gpu.nvidia.com` device class. The RayJob references this template by name. Each cloud overlay applies JSON patches to place pods on the correct node pools:

| Pod | Azure | Nebius |
|---|---|---|
| Submitter / Head | `agentpool: cpu` | `agentpool: nebius-cpu` |
| GPU Workers | `agentpool: gpu` | `agentpool: nebius-gpu` |

## Architecture

```
./run.sh azure   (or: kubectl apply -k overlays/azure)
        │
        ▼
┌─────────────────────────────────────────────────┐
│ RayJob: multimodel-distributed-training         │
│                                                 │
│  Head Pod (CPU node pool)                       │
│  ├── main.py (entrypoint)                       │
│  ├── doggos/embed.py (CLIP actor)               │
│  └── /mnt/cluster_storage (shared PVC)          │
│                                                 │
│  Worker Pods (GPU) × 2                          │
│  ├── Ray Train workers (DDP)                    │
│  ├── Ray Data actors (CLIP embedding)           │
│  └── /mnt/cluster_storage (shared PVC)          │
└─────────────────────────────────────────────────┘
```

The training script is mounted via a ConfigMap. Pip dependencies (`torch`, `transformers`, `mlflow`, `doggos`, etc.) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`. The `doggos` helper package is installed directly from [GitHub](https://github.com/anyscale/multimodal-ai).

## Quick Start

### 1. Ensure PVC and StorageClass exist

```bash
kubectl apply -f configs/storageclass.yaml
kubectl apply -f configs/pvc.yaml
```

### 2. Run the example

The `run.sh` script handles ConfigMap creation, cleanup of previous runs, KubeRay operator installation (if needed), and applies the correct kustomize overlay:

```bash
./run.sh azure    # for AKS clusters
./run.sh nebius   # for Nebius clusters
```

Or apply manually:

```bash
# Create the ConfigMap
kubectl create configmap multimodel-distributed-training-scripts \
    --from-file=main.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -

# Apply the overlay
kubectl apply -k overlays/azure   # or overlays/nebius
```

### 3. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob multimodel-distributed-training -w

# Stream logs
kubectl -n ray logs -f -l job-name=multimodel-distributed-training --tail=100

# Ray Dashboard
kubectl -n ray port-forward svc/multimodel-distributed-training-head-svc 8265:8265
```

### 4. View MLflow Metrics

```bash
# Run MLflow server inside the head pod
head_pod=$(kubectl get pod -l ray.io/node-type=head -o=jsonpath='{.items[0].metadata.name}' -n ray)
kubectl -n ray exec -it $head_pod -- \
    mlflow server -h 0.0.0.0 -p 8080 \
    --backend-store-uri /mnt/cluster_storage/mlflow/doggos

# Port-forward in another terminal
kubectl -n ray port-forward $head_pod 8080:8080
```

MLflow UI is now available at [http://localhost:8080](http://localhost:8080).

## Adapting the Scaling Config

The default `base/rayjob.yaml` is configured for **2 GPU worker nodes with 8 GPUs each** (e.g., `Standard_ND96asr_v4`). Adjust for your setup:

| Node Pool VM SKU | GPUs/Node | `NUM_WORKERS` | `resources_per_worker` |
|---|---|---|---|
| `Standard_NC6s_v3` (V100) | 1 | 2 | `{"CPU": 4, "GPU": 1}` |
| `Standard_NC24ads_A100_v4` (A100) | 4 | 2 | `{"CPU": 8, "GPU": 4}` |
| `Standard_ND96asr_v4` (A100 x8) | 8 | 2 | `{"CPU": 8, "GPU": 8}` |

Update both `base/rayjob.yaml` (worker replicas) and the GPU `count` in `base/gpu-claim.yaml`, along with the constants at the top of `main.py`.

## Key Differences from Anyscale Version

| Anyscale | KubeRay on AKS / Nebius |
|---|---|
| Notebook runs inside Anyscale Workspace | Script runs on the cluster via RayJob |
| `accelerator_type="T4"` | Removed — GPU type determined by VM SKU |
| S3 user storage for artifacts | Shared PVC at `/mnt/cluster_storage` |
| Anyscale runtime env auto-setup | `runtimeEnvYAML` in RayJob spec |
| `anyscale job submit` | `./run.sh azure` or `kubectl apply -k overlays/<cloud>` |

## Cleanup

```bash
kubectl -n ray delete rayjob multimodel-distributed-training
kubectl -n ray delete configmap multimodel-distributed-training-scripts
```
