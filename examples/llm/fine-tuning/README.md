# LLM Fine-Tuning on AKS with KubeRay

This example demonstrates **LLM fine-tuning for entity recognition** on Kubernetes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Train](https://docs.ray.io/en/latest/train/train.html) with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). It supports both **Azure (AKS)** and **Nebius** clusters.

## What This Example Does

Runs an end-to-end entity recognition pipeline (training + batch inference) using Ray Train with distributed GPU workers. The script uses Ray object store to distribute dataset and checkpoint files between head and GPU worker nodes — no shared PVC is required.

## Prerequisites

| Component | Version / Details |
|---|---|
| Kubernetes cluster | AKS or Nebius with GPU node pool |
| NVIDIA GPU DRA driver | `gpu.nvidia.com` device class available on GPU nodes |
| KubeRay operator | v1.5.1 (installed by `run.sh` if not present) |
| Ray | 2.53.0 |

## Directory Structure

```
fine-tuning/
├── main.py                          # Fine-tuning script (runs on the RayCluster)
├── run.sh                           # One-command launcher (azure or nebius)
├── base/
│   ├── kustomization.yaml           # Kustomize base
│   ├── rayjob.yaml                  # Cloud-agnostic RayJob manifest
│   └── gpu-claim.yaml               # DRA ResourceClaimTemplate (8 GPUs per worker)
├── overlays/
│   ├── azure/
│   │   ├── kustomization.yaml       # Azure overlay
│   │   └── rayjob-patch.yaml        # nodeSelector for Azure
│   └── nebius/
│       ├── kustomization.yaml       # Nebius overlay
│       └── rayjob-patch.yaml        # nodeSelector for Nebius
├── resourceclaim.yaml               # DRA reference (example allocated claim)
├── resourceclaimtemplate.yaml        # DRA reference (standalone template)
└── resourceslice.yaml               # DRA reference (node GPU inventory)
```

GPU allocation is defined in `base/gpu-claim.yaml` as a standalone [ResourceClaimTemplate](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) (`multi-gpu`) that requests 8 NVIDIA H100 GPUs per worker via the `gpu.nvidia.com` device class. The RayJob references this template by name. Each cloud overlay applies JSON patches to place pods on the correct node pools:

| Pod | Azure | Nebius |
|---|---|---|
| Submitter / Head | `agentpool: cpu` | `agentpool: nebius-cpu` |
| GPU Workers | `agentpool: gpu` | `agentpool: nebius-gpu` |

> **Note:** The `resourceclaim*.yaml` and `resourceslice.yaml` files are reference manifests showing what DRA objects look like when allocated on a live cluster. They are not applied directly — the RayJob references the standalone `ResourceClaimTemplate` defined in `base/gpu-claim.yaml`.

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

The default `base/rayjob.yaml` uses **2 GPU worker nodes with 8 H100 GPUs each** (16 total), allocated via DRA. Adjust `replicas`, `num-gpus` in `base/rayjob.yaml` and the GPU `count` in `base/gpu-claim.yaml` for your node pool configuration.

## Cleanup

```bash
kubectl -n ray delete rayjob llm-fine-tuning
kubectl -n ray delete configmap llm-fine-tuning-scripts
```
