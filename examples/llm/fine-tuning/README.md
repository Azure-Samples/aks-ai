# LLM Fine-Tuning on AKS with KubeRay

This example demonstrates **LLM fine-tuning for entity recognition** on Kubernetes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Train](https://docs.ray.io/en/latest/train/train.html) with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). It supports both **Azure (AKS)** and **Nebius** clusters.

## What This Example Does

Runs an end-to-end entity recognition pipeline (training + batch inference) using Ray Train with distributed GPU workers. The script uses Ray object store to distribute dataset and checkpoint files between head and GPU worker nodes — no shared PVC is required.

## Prerequisites

| Component | Version / Details |
|---|---|
| Kubernetes cluster | AKS or Nebius with GPU node pool |
| NVIDIA device plugin | Installed via cluster setup |
| KubeRay operator | v1.5.1 (installed by `run.sh` if not present) |
| Ray | 2.53.0 |

## Directory Structure

```
fine-tuning/
├── main.py                          # Fine-tuning script (runs on the RayCluster)
├── run.sh                           # One-command launcher (azure or nebius)
├── base/
│   ├── kustomization.yaml           # Kustomize base
│   └── rayjob.yaml                  # Cloud-agnostic RayJob manifest
├── overlays/
│   ├── azure/
│   │   ├── kustomization.yaml       # Azure overlay
│   │   └── rayjob-patch.yaml        # nodeSelector + tolerations for Azure
│   └── nebius/
│       ├── kustomization.yaml       # Nebius overlay
│       └── rayjob-patch.yaml        # nodeSelector for Nebius
├── resourceclaim.yaml               # DRA reference (for clusters using DRA)
├── resourceclaimtemplate.yaml        # DRA reference
└── resourceslice.yaml               # DRA reference
```

The `base/rayjob.yaml` contains no hardcoded scheduling. Each cloud overlay applies JSON patches to place pods on the correct node pools:

| Pod | Azure | Nebius |
|---|---|---|
| Submitter / Head | `agentpool: cpu` | `agentpool: nebius-cpu` |
| GPU Workers | `agentpool: gpu` + `sku=gpu:NoSchedule` toleration | `agentpool: nebius-gpu` |

> **Note:** The `resourceclaim*.yaml` and `resourceslice.yaml` files are reference manifests for clusters using Kubernetes Dynamic Resource Allocation (DRA) instead of standard `nvidia.com/gpu` resource limits. They are not used by the kustomize overlays.

## Architecture

```
./run.sh azure   (or: kubectl apply -k overlays/azure)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ RayJob: llm-fine-tuning                          │
│                                                  │
│  Head Pod (CPU node pool)                        │
│  ├── main.py (entrypoint via ConfigMap)          │
│  └── Coordinates training                        │
│                                                  │
│  Worker Pods (GPU) × 2                           │
│  ├── 8 GPUs each (16 total)                      │
│  └── Ray Train workers (DDP)                     │
└──────────────────────────────────────────────────┘
```

The script is mounted via a ConfigMap. Pip dependencies (`llamafactory`, `pynvml`, `tensorboard`, etc.) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`.

## Quick Start

### 1. Run the example

The `run.sh` script handles ConfigMap creation, cleanup of previous runs, KubeRay operator installation (if needed), and applies the correct kustomize overlay:

```bash
./run.sh azure    # for AKS clusters
./run.sh nebius   # for Nebius clusters
```

Or apply manually:

```bash
# Create the ConfigMap
kubectl create configmap llm-fine-tuning-scripts \
    --from-file=main.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -

# Apply the overlay
kubectl apply -k overlays/azure   # or overlays/nebius
```

### 2. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob llm-fine-tuning -w

# Stream logs
kubectl -n ray logs -f -l job-name=llm-fine-tuning --tail=100

# Ray Dashboard
kubectl -n ray port-forward svc/llm-fine-tuning-head-svc 8265:8265
```

## Scaling

The default `base/rayjob.yaml` uses **2 GPU worker nodes with 8 GPUs each** (16 total). Adjust `replicas`, `num-gpus`, and `nvidia.com/gpu` values in `base/rayjob.yaml` for your node pool configuration.

## Cleanup

```bash
kubectl -n ray delete rayjob llm-fine-tuning
kubectl -n ray delete configmap llm-fine-tuning-scripts
```
