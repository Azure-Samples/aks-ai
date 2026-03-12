# LLM Distributed Inference Benchmark on AKS with KubeRay

This example demonstrates **LLM inference benchmarking** on Kubernetes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Data LLM](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.html) backed by [vLLM](https://docs.vllm.ai/). It supports both **Azure (AKS)** and **Nebius** clusters.

## What This Example Does

1. **Build prompts** — Generates a set of prompts from built-in templates.
2. **Preprocess** — Converts prompts into chat-format messages for the model.
3. **Inference** — Runs distributed inference through vLLM with tensor parallelism via Ray Data.
4. **Measure** — Collects throughput and latency metrics from the results.

## Prerequisites

| Component | Version / Details |
|---|---|
| Kubernetes cluster | AKS or Nebius with GPU node pool |
| NVIDIA device plugin | Installed via cluster setup |
| KubeRay operator | v1.5.1+, installed via cluster setup |
| Ray | 2.53.0 |

## Directory Structure

```
distributed-inferencing/
├── main.py                          # Benchmark script (runs on the RayCluster)
├── run.sh                           # One-command launcher (azure or nebius)
├── base/
│   ├── kustomization.yaml           # Kustomize base
│   └── rayjob.yaml                  # Cloud-agnostic RayJob manifest
└── overlays/
    ├── azure/
    │   ├── kustomization.yaml       # Azure overlay
    │   └── rayjob-patch.yaml        # nodeSelector + tolerations for Azure
    └── nebius/
        ├── kustomization.yaml       # Nebius overlay
        └── rayjob-patch.yaml        # nodeSelector for Nebius
```

The `base/rayjob.yaml` contains no hardcoded scheduling. Each cloud overlay applies JSON patches to place pods on the correct node pools:

| Pod | Azure | Nebius |
|---|---|---|
| Submitter / Head | `agentpool: cpu` | `agentpool: nebius-cpu` |
| GPU Workers | `agentpool: gpu` + `sku=gpu:NoSchedule` toleration | `agentpool: nebius-gpu` |

## Architecture

```
./run.sh azure   (or: kubectl apply -k overlays/azure)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ RayJob: llm-distributed-inferencing              │
│                                                  │
│  Head Pod (CPU node pool)                        │
│  ├── main.py (entrypoint via ConfigMap)          │
│  └── Coordinates the inference pipeline          │
│                                                  │
│  Worker Pods (GPU) × 2                           │
│  ├── 1 GPU each (2 total)                        │
│  ├── vLLM engine via ray.data.llm                │
│  └── HuggingFace model cache (emptyDir)          │
└──────────────────────────────────────────────────┘
```

The script is mounted via a ConfigMap. Pip dependencies (`ray[llm]`, `vllm`) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`. Workers mount an `emptyDir` volume at `/root/.cache/huggingface` for model weight caching.

## Quick Start

### 1. Run the example

The `run.sh` script handles ConfigMap creation, cleanup of previous runs, and applies the correct kustomize overlay:

```bash
./run.sh azure    # for AKS clusters
./run.sh nebius   # for Nebius clusters
```

Or apply manually:

```bash
# Create the ConfigMap
kubectl create configmap llm-distributed-inferencing-scripts \
    --from-file=main.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -

# Apply the overlay
kubectl apply -k overlays/azure   # or overlays/nebius
```

### 2. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob llm-distributed-inferencing -w

# Stream logs
kubectl -n ray logs -f -l job-name=llm-distributed-inferencing --tail=200

# Ray Dashboard
kubectl -n ray port-forward svc/llm-distributed-inferencing-head-svc 8265:8265
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `NUM_PROMPTS` | `50` | Number of prompts to benchmark |
| `MAX_TOKENS` | `256` | Maximum tokens per response |
| `TENSOR_PARALLEL_SIZE` | `1` | Tensor parallelism degree for vLLM |
| `CONCURRENCY` | `1` | Number of concurrent engine replicas |

These are set in `runtimeEnvYAML` inside `base/rayjob.yaml` and can be overridden there.

### Scaling

The default `base/rayjob.yaml` uses **2 GPU worker nodes with 1 GPU each** (2 total). Adjust `replicas`, `num-gpus`, and `nvidia.com/gpu` values in `base/rayjob.yaml` and the `TENSOR_PARALLEL_SIZE` / `CONCURRENCY` environment variables for your node pool configuration.

## Cleanup

```bash
kubectl -n ray delete rayjob llm-distributed-inferencing
kubectl -n ray delete configmap llm-distributed-inferencing-scripts
```
