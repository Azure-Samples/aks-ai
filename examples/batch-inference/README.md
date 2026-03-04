# Batch Inference on AKS with KubeRay

This example demonstrates **batch inference** (CLIP image embedding generation) on Azure Kubernetes Service (AKS) using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and [Ray Data](https://docs.ray.io/en/latest/data/data.html).

It is adapted from the [Ray E2E Multimodal AI Workloads — Batch Inference](https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/01-Batch-Inference.html) tutorial (originally designed for Anyscale) to run on AKS.

## What This Example Does

1. **Distributed Read** — Reads dog breed images from a public S3 bucket using Ray Data (CPU).
2. **Preprocessing** — Adds class labels extracted from file paths using `map` (CPU).
3. **Batch Embedding** — Generates CLIP embeddings with GPU actors via `map_batches` (GPU).
4. **Distributed Write** — Writes embeddings to shared storage as Parquet (CPU).
5. **Similarity Search** — Embeds a query image and retrieves the most similar images by cosine similarity.

The pipeline uses Ray Data's **streaming execution**, which processes data in chunks as they're loaded — avoiding OOM errors on large datasets and maximizing GPU utilization by overlapping CPU preprocessing with GPU inference.

```
S3 (images) ──► read_images (CPU) ──► map(add_class) (CPU)
                                           │
                                           ▼
                        map_batches(EmbedImages) (GPU × N)
                                           │
                                           ▼
                           write_parquet (CPU) ──► Shared Storage
```

## Prerequisites

| Component | Version / Details |
|---|---|
| AKS cluster | Created via `scripts/setup.sh` with GPU node pool |
| NVIDIA device plugin | Installed via cluster setup |
| KubeRay operator | v1.5.1+, installed via cluster setup |
| Shared PVC | `cluster-storage` (ReadWriteMany), created via `configs/pvc.yaml` |
| Ray | 2.48.0 |

## Files

| File | Description |
|---|---|
| `batch_inference.py` | Batch inference script — runs on the RayCluster |
| `rayjob.yaml` | RayJob manifest — submits the script and manages the cluster |
| `requirements.txt` | Python dependencies |

## Architecture

```
kubectl apply -f rayjob.yaml
        │
        ▼
┌──────────────────────────────────────────────────┐
│ RayJob: batch-inference                          │
│                                                  │
│  Head Pod (CPU, system node pool)                │
│  ├── batch_inference.py (entrypoint)             │
│  └── /mnt/cluster_storage (Azure Blob PVC)       │
│                                                  │
│  Worker Pods (GPU) × 1                           │
│  ├── Ray Data actors (CLIP embedding, GPU)       │
│  └── /mnt/cluster_storage (Azure Blob PVC)       │
└──────────────────────────────────────────────────┘
```

The script is mounted via a ConfigMap. Pip dependencies (`torch`, `transformers`, `scipy`, etc.) are installed on all nodes at job start via `runtimeEnvYAML` in `rayjob.yaml`.

## Quick Start

### 1. Ensure PVC and StorageClass exist

```bash
kubectl apply -f configs/storageclass.yaml
kubectl apply -f configs/pvc.yaml
```

### 2. Create the ConfigMap

Package the batch inference script so the RayJob pods can access it:

```bash
kubectl create configmap batch-inference-scripts \
    --from-file=batch_inference.py \
    -n ray --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Submit the RayJob

```bash
kubectl apply -f rayjob.yaml
```

This creates a RayCluster (head + 1 GPU worker), installs pip dependencies via `runtimeEnvYAML`, runs `batch_inference.py`, and keeps the cluster alive for inspection.

### 4. Monitor

```bash
# Watch job status
kubectl -n ray get rayjob batch-inference -w

# Stream logs
kubectl -n ray logs -f -l job-name=batch-inference --tail=100

# Ray Dashboard
kubectl -n ray port-forward svc/batch-inference-head-svc 8265:8265
```

Then open [http://localhost:8265](http://localhost:8265) for the Ray Dashboard.

### 5. Inspect Results

After the job completes, the embeddings are stored at `/mnt/cluster_storage/doggos/embeddings` on the PVC. You can inspect them from any pod with the PVC mounted:

```bash
head_pod=$(kubectl get pod -l ray.io/node-type=head -o=jsonpath='{.items[0].metadata.name}' -n ray)
kubectl -n ray exec -it $head_pod -- python -c "
import ray
ray.init()
ds = ray.data.read_parquet('/mnt/cluster_storage/doggos/embeddings')
print(ds.schema())
print(f'Total rows: {ds.count()}')
print(ds.take(3))
"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDINGS_DIR` | `/mnt/cluster_storage/doggos/embeddings` | Output directory for Parquet files |
| `BATCH_SIZE` | `64` | Batch size for CLIP embedding |
| `NUM_GPU_ACTORS` | `4` | Number of GPU actor replicas |
| `TOP_K` | `5` | Number of similar images to retrieve |
| `SAMPLE_IMAGE_URL` | `https://doggos-dataset.s3...samara.png` | Query image for similarity demo |

These are set in `runtimeEnvYAML` inside `rayjob.yaml` and can be overridden there.

### Scaling

The default `rayjob.yaml` uses **1 GPU worker node with 8 GPUs** and 4 CLIP embedding actors. Adjust for your setup:

| Node Pool VM SKU | GPUs/Node | Suggested `NUM_GPU_ACTORS` | Worker Replicas |
|---|---|---|---|
| `Standard_NC6s_v3` (V100) | 1 | 1 | 1 |
| `Standard_NC24ads_A100_v4` (A100) | 1 | 1 | 4 |
| `Standard_ND96asr_v4` (A100 x8) | 8 | 4 | 1 |
| `gpu-h100-sxm-8gpu` (H100 x8) | 8 | 4 | 1 |

Update the `replicas`, `num-gpus`, and `nvidia.com/gpu` values in `rayjob.yaml` along with the `NUM_GPU_ACTORS` environment variable.

## Key Differences from Anyscale Version

| Anyscale | AKS + KubeRay |
|---|---|
| Notebook runs inside Anyscale Workspace | Script runs on the cluster via RayJob |
| `accelerator_type="T4"` | Removed — GPU type determined by VM SKU |
| S3 user storage for artifacts | Azure Blob PVC at `/mnt/cluster_storage` |
| Anyscale runtime env auto-setup | `runtimeEnvYAML` in RayJob spec |
| `anyscale job submit` | `kubectl apply -f rayjob.yaml` |
| `doggos` pip package from GitHub | Self-contained `EmbedImages` class in script |

## Cleanup

```bash
kubectl delete -f rayjob.yaml
kubectl delete configmap batch-inference-scripts -n ray
```
