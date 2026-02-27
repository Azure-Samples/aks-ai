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

### 1. Create the ConfigMap

Package the training script so the RayJob pods can access it:

```bash
kubectl create configmap distributed-training-scripts -n ray \
    --from-file=distributed_training.py=distributed_training.py
```

### 2. Ensure PVC and StorageClass exist

```bash
kubectl apply -f configs/storageclass.yaml
kubectl apply -f configs/pvc.yaml
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
kubectl -n ray port-forward svc/distributed-training-raycluster-head-svc 8265:8265
```

### 5. View MLflow Metrics

```bash
# Run MLflow server inside the head pod
kubectl -n ray exec -it <head-pod> -- \
    mlflow server -h 0.0.0.0 -p 8080 \
    --backend-store-uri /mnt/cluster_storage/mlflow/doggos

# Port-forward in another terminal
kubectl -n ray port-forward <head-pod> 8080:8080
```

MLflow UI is now available at [http://localhost:8080](http://localhost:8080).

## Code Walkthrough

### Preprocessing

The script reads dog breed images from a public S3 bucket and converts them into CLIP embeddings using GPU actors:

```python
# doggos/embed.py — runs on GPU workers via ray.data.map_batches
class EmbedImages:
    def __init__(self, model_id, device="cuda"):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.device = device

    def __call__(self, batch):
        images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in batch["image"]]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            batch["embedding"] = self.model.get_image_features(**inputs).cpu().numpy()
        return batch
```

The `Preprocessor` class fits a class-to-label mapping and transforms datasets:

```python
train_ds = ray.data.read_images("s3://doggos-dataset/train", include_paths=True, shuffle="files")
train_ds = train_ds.map(add_class)

preprocessor = Preprocessor()
preprocessor.fit(train_ds, column="class")
train_ds = preprocessor.transform(ds=train_ds)  # Runs EmbedImages on GPU actors

# Write to shared PVC to avoid recomputing
train_ds.write_parquet("/mnt/cluster_storage/doggos/preprocessed_data/preprocessed_train")
```

> **AKS Note**: The original Anyscale example uses `accelerator_type="T4"`. This is removed — on AKS, GPU types are determined by the VM SKU in your node pool.

### Model

A simple two-layer neural net with BatchNorm and Dropout. Pure PyTorch — no Ray-specific code in the model itself:

```python
class ClassificationModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_p, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        z = self.fc1(batch["embedding"])
        z = self.batch_norm(z)
        z = self.relu(z)
        z = self.dropout(z)
        return self.fc2(z)
```

### Distributed Training

Ray Train wraps the PyTorch training loop with minimal changes:

- `ray.train.torch.prepare_model(model)` — wraps with DDP
- `ray.train.get_dataset_shard("train")` — each worker gets its shard
- `ScalingConfig` — defines workers and GPUs

```python
# Scaling config — adjust based on your GPU node pool
SCALING_CONFIG = ray.train.ScalingConfig(
    num_workers=4,
    use_gpu=True,
    resources_per_worker={"CPU": 8, "GPU": 2},
)

def train_loop_per_worker(config):
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    model = ClassificationModel(...)
    model = ray.train.torch.prepare_model(model)  # Wrap with DDP

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(train_ds, ...)
        val_loss, _, _ = eval_epoch(val_ds, ...)

        # MLflow logging (rank 0 only)
        if ray.train.get_context().get_world_rank() == 0:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss})

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=SCALING_CONFIG,
    datasets={"train": preprocessed_train_ds, "val": preprocessed_val_ds},
)
results = trainer.fit()
```

### Evaluation

After training, the script loads the best model from MLflow and runs batch inference on the test set:

```python
predictor = TorchPredictor.from_artifacts_dir(artifacts_dir)
test_ds = ray.data.read_images("s3://doggos-dataset/test", include_paths=True)
test_ds = test_ds.map(add_class)
test_ds = predictor.preprocessor.transform(ds=test_ds)

pred_ds = test_ds.map_batches(predictor, concurrency=4, batch_size=64, num_gpus=1)
```

Reports precision, recall, F1, and accuracy.

## Adapting the Scaling Config

The default `rayjob.yaml` is configured for **2 GPU worker nodes with 4 GPUs each** (e.g., `Standard_NC24ads_A100_v4`). Adjust for your setup:

| Node Pool VM SKU | GPUs/Node | `NUM_WORKERS` | `resources_per_worker` |
|---|---|---|---|
| `Standard_NC6s_v3` (V100) | 1 | 2 | `{"CPU": 4, "GPU": 1}` |
| `Standard_NC24ads_A100_v4` (A100) | 4 | 4 | `{"CPU": 8, "GPU": 2}` |
| `Standard_ND96asr_v4` (A100 x8) | 8 | 8 | `{"CPU": 8, "GPU": 1}` |

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
