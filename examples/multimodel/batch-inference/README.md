# Batch Inference on AKS with KubeRay

This example demonstrates **batch inference** (CLIP image embedding generation) on Azure Kubernetes Service (AKS) using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Data](https://docs.ray.io/en/latest/data/data.html).

It is adapted from the [Ray E2E Multimodal AI Workloads — Batch Inference](https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/01-Batch-Inference.html) tutorial (originally designed for Anyscale) to run on AKS.

## What This Example Does

1. **Distributed Read** — Reads dog breed images from a public S3 bucket using Ray Data (CPU).
2. **Preprocessing** — Adds class labels extracted from file paths using `map` (CPU).
3. **Batch Embedding** — Generates CLIP embeddings with GPU actors via `map_batches` (GPU).
4. **Materialize** — Materializes embeddings into Ray's shared memory object store.
5. **Similarity Search** — Embeds a query image and retrieves the most similar images by cosine similarity.

The pipeline uses Ray Data's **streaming execution**, which processes data in chunks as they're loaded — avoiding OOM errors on large datasets and maximizing GPU utilization by overlapping CPU preprocessing with GPU inference.

```
S3 (images) ──► read_images (CPU) ──► map(add_class) (CPU)
                                           │
                                           ▼
                        map_batches(EmbedImages) (GPU × N)
                                           │
                                           ▼
                              materialize() ──► Ray Object Store
```

## Prerequisites

| Component | Version / Details |
|---|---|
| AKS cluster | Created via `scripts/setup.sh` with GPU node pool |
| NVIDIA device plugin | Installed via cluster setup |
| KubeRay operator | v1.5.1+, installed via cluster setup |
| Ray | 2.48.0 |

## Files

| File | Description |
|---|---|
| `main.py` | Batch inference script — runs on the RayCluster |
| `rayjob.yaml` | RayJob manifest — submits the script and manages the cluster |

## Architecture

```
kubectl apply -f rayjob.yaml
        │
        ▼
┌──────────────────────────────────────────────────┐
│ RayJob: batch-inference                          │
│                                                  │
│  Head Pod (CPU, system node pool)                │
│  ├── main.py (entrypoint via ConfigMap)            │
│  └── Drives the Ray Data pipeline                │
│                                                  │
│  Worker Pods (GPU) × 2                           │
│  ├── 1 GPU each (2 total)                        │
│  └── CLIP embedding actors via map_batches       │
│                                                  │
│  Ray Object Store (shared memory)                │
│  └── Materialized embeddings (ephemeral)         │
└──────────────────────────────────────────────────┘
```

The script is mounted via a ConfigMap. Pip dependencies (`torch`, `transformers`, `doggos`, etc.) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`.

## Quick Start

### 1. Create the ConfigMap

Package the batch inference script so the RayJob pods can access it:

```bash
kubectl create configmap batch-inference-scripts \
    --from-file=main.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Submit the RayJob

```bash
kubectl apply -f rayjob.yaml
```

This creates a RayCluster (head + 2 GPU workers with 1 GPU each), installs pip dependencies via `runtimeEnvYAML`, runs `main.py`, and keeps the cluster alive for inspection.

### 3. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob batch-inference -w

# Stream logs
kubectl -n ray logs -f -l job-name=batch-inference --tail=100

# Ray Dashboard
kubectl -n ray port-forward svc/batch-inference-head-svc 8265:8265
```

Then open [http://localhost:8265](http://localhost:8265) for the Ray Dashboard.

### 4. Review Output

Embeddings are materialized into Ray's in-memory object store and used directly for the similarity search. The top-K results are printed in the job logs:

```
Top 5 similar images:
  1. class=border_collie        similarity=0.8176  path=s3://...
  2. class=yorkshire_terrier    similarity=0.8079  path=s3://...
  ...
```

Since embeddings live in Ray's object store, they are **ephemeral** — they exist only while the RayCluster is running. Set `shutdownAfterJobFinishes: false` (the default in `rayjob.yaml`) to keep the cluster alive for interactive inspection via the Ray Dashboard.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `64` | Batch size for CLIP embedding |
| `NUM_GPU_ACTORS` | `4` | Number of GPU actor replicas (set to `2` in `rayjob.yaml`) |
| `TOP_K` | `5` | Number of similar images to retrieve |
| `SAMPLE_IMAGE_URL` | `https://doggos-dataset.s3...samara.png` | Query image for similarity demo |

These are set in `runtimeEnvYAML` inside `rayjob.yaml` and can be overridden there.

### Scaling

The default `rayjob.yaml` uses **2 GPU worker nodes with 1 GPU each** (2 total) and 2 CLIP embedding actors. Adjust for your setup:

| Node Pool VM SKU | GPUs/Node | Suggested `NUM_GPU_ACTORS` | Worker Replicas |
|---|---|---|---|
| `Standard_NC6s_v3` (V100) | 1 | 1 | 1 |
| `Standard_NC24ads_A100_v4` (A100) | 1 | 1 | 4 |
| `Standard_ND96asr_v4` (A100 x8) | 8 | 8 | 1 |
| `gpu-h100-sxm-8gpu` (H100 x8) | 8 | 8–16 | 1–2 |

Update the `replicas`, `num-gpus`, and `nvidia.com/gpu` values in `rayjob.yaml` along with the `NUM_GPU_ACTORS` environment variable.

> **Note:** The `runtimeEnvYAML` pip install runs per-actor on each worker node at startup. With large dependencies like `torch` (~2.8 GB), expect a 1–2 minute delay before GPU actors begin processing. To eliminate this delay, bake dependencies into a custom container image.

## Key Differences from Anyscale Version

| Anyscale | AKS + KubeRay |
|---|---|
| Notebook runs inside Anyscale Workspace | Script runs on the cluster via RayJob |
| `accelerator_type="T4"` | Removed — GPU type determined by VM SKU |
| S3 user storage for Parquet artifacts | Ray object store via `materialize()` (ephemeral) |
| Anyscale runtime env auto-setup | `runtimeEnvYAML` in RayJob spec |
| `anyscale job submit` | `kubectl apply -f rayjob.yaml` |
| `doggos` pip package pre-installed | `doggos` installed via `runtimeEnvYAML` from GitHub |

## Cleanup

```bash
kubectl delete -f rayjob.yaml
kubectl delete configmap batch-inference-scripts -n ray
```
