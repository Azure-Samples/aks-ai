"""
Distributed Training on AKS with KubeRay

Adapted from: https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/02-Distributed-Training.html

This script runs on the RayCluster (submitted via RayJob). It:
  1. Preprocesses dog breed images using CLIP embeddings (Ray Data + GPU actors)
  2. Trains a classifier with Ray Train TorchTrainer (distributed DDP)
  3. Logs metrics to MLflow, saves best model via Ray object store
  4. Evaluates the model on a held-out test set
"""

import gc
import os
import ray
import pyarrow.fs

import numpy as np
import torch
from doggos.embed import EmbedImages

import shutil

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.train.torch import get_device

import tempfile
import mlflow
from ray.train.torch import TorchTrainer

from sklearn.metrics import multilabel_confusion_matrix


# ---------------------------------------------------------------------------
# Model store – passes best model from training workers to driver via Ray
# ---------------------------------------------------------------------------
@ray.remote
class _ModelStore:
    def __init__(self):
        self._artifacts = None

    def put(self, artifacts):
        self._artifacts = artifacts

    def get(self):
        return self._artifacts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "doggos"
TRAIN_LOOP_CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "embedding_dim": 512,
    "hidden_dim": 256,
    "dropout_p": 0.3,
    "lr": 1e-3,
    "lr_factor": 0.8,
    "lr_patience": 3,
    "num_epochs": 20,
    "batch_size": 256,
}

# Adjust num_workers and resources_per_worker based on your GPU node pool.
NUM_WORKERS = 16
SCALING_CONFIG = ray.train.ScalingConfig(
    num_workers=NUM_WORKERS,
    use_gpu=True,
    resources_per_worker={"CPU": 1, "GPU": 1},
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def add_class(row):
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


def convert_to_label(row, class_to_label):
    if "class" in row:
        row["label"] = class_to_label[row["class"]]
    return row


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
class Preprocessor:
    """Preprocessor class."""

    def __init__(self, class_to_label=None):
        self.class_to_label = class_to_label or {}  # mutable defaults
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}

    def fit(self, ds, column):
        self.classes = ds.unique(column=column)
        self.class_to_label = {tag: i for i, tag in enumerate(self.classes)}
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        return self

    def transform(self, ds, concurrency=4, batch_size=64, num_gpus=1):
        ds = ds.map(
            convert_to_label,
            fn_kwargs={"class_to_label": self.class_to_label},
        )
        ds = ds.map_batches(
            EmbedImages,
            fn_constructor_kwargs={
                "model_id": "openai/clip-vit-base-patch32",
                "device": "cuda",
            },
            concurrency=concurrency,
            batch_size=batch_size,
            num_gpus=num_gpus,
        )
        ds = ds.drop_columns(["image"])
        return ds

    def save(self, fp):
        with open(fp, "w") as f:
            json.dump(self.class_to_label, f)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ClassificationModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_p, num_classes):
        super().__init__()
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_classes = num_classes

        # Define layers
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
        z = self.fc2(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_probabilities(self, batch):
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        Path(dp).mkdir(parents=True, exist_ok=True)
        with open(Path(dp, "args.json"), "w") as fp:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout_p": self.dropout_p,
                    "num_classes": self.num_classes,
                },
                fp,
                indent=4,
            )
        torch.save(self.state_dict(), Path(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp, device="cpu"):
        with open(args_fp, "r") as fp:
            model = cls(**json.load(fp))
        model.load_state_dict(torch.load(state_dict_fp, map_location=device))
        return model


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def collate_fn(batch, device=None):
    dtypes = {"embedding": torch.float32, "label": torch.int64}
    tensor_batch = {}

    # If no device is provided, try to get it from Ray Train context
    if device is None:
        try:
            device = get_device()
        except RuntimeError:
            # When not in Ray Train context, use CPU for testing
            device = "cpu"

    for key in dtypes.keys():
        if key in batch:
            tensor_batch[key] = torch.as_tensor(
                batch[key],
                dtype=dtypes[key],
                device=device,
            )
    return tensor_batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(ds, batch_size, model, num_classes, loss_fn, optimizer):
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # Reset gradients
        z = model(batch)  # Forward pass
        targets = F.one_hot(batch["label"], num_classes=num_classes).float()
        J = loss_fn(z, targets)  # Define loss
        J.backward()  # Backward pass
        optimizer.step()  # Update weights
        loss += (J.detach().item() - loss) / (i + 1)  # Cumulative loss
    return loss


def eval_epoch(ds, batch_size, model, num_classes, loss_fn):
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(
                batch["label"], num_classes=num_classes
            ).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["label"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config):
    # Hyperparameters
    experiment_name = config["experiment_name"]
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # MLflow tracking (rank 0 only).
    # Use a LOCAL temp dir for the MLflow tracking URI so that all internal
    # MLflow file operations (append-mode metric writes, metadata updates)
    # happen on a real POSIX filesystem.  Mountpoint S3 does NOT support
    # open("a") (append), which MLflow FileStore uses for log_metrics().
    # After training we copy the entire experiment tree to the S3 mount.
    local_mlflow_dir = None
    if ray.train.get_context().get_world_rank() == 0:
        local_mlflow_dir = tempfile.mkdtemp(prefix="mlflow_local_")
        mlflow.set_tracking_uri(f"file:{local_mlflow_dir}")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(config)

    # Datasets.
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    # Model.
    model = ClassificationModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        num_classes=num_classes,
    )
    model = ray.train.torch.prepare_model(model)

    # Training components.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    # Training.
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(num_epochs):
        # Steps
        train_loss = train_epoch(
            train_ds, batch_size, model, num_classes, loss_fn, optimizer
        )
        val_loss, _, _ = eval_epoch(val_ds, batch_size, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        metrics = dict(
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            val_loss=val_loss,
        )
        if ray.train.get_context().get_world_rank() == 0:
            mlflow.log_metrics(metrics, step=epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "state_dict": {k: v.cpu() for k, v in model.module.state_dict().items()},
                    "args": {
                        "embedding_dim": embedding_dim,
                        "hidden_dim": hidden_dim,
                        "dropout_p": dropout_p,
                        "num_classes": num_classes,
                    },
                    "class_to_label": config["class_to_label"],
                }
        ray.train.report(metrics)

    # Save best model to the shared actor and end MLflow tracking.
    if ray.train.get_context().get_world_rank() == 0:
        store = ray.get_actor("model_store")
        ray.get(store.put.remote(best_state))
        mlflow.end_run()
        shutil.rmtree(local_mlflow_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class TorchPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch, device="cuda"):
        self.model.to(device)
        batch["prediction"] = self.model.predict(collate_fn(batch, device=device))
        return batch

    def predict_probabilities(self, batch, device="cuda"):
        self.model.to(device)
        predicted_probabilities = self.model.predict_probabilities(
            collate_fn(batch, device=device)
        )
        batch["probabilities"] = [
            {
                self.preprocessor.label_to_class[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            for probabilities in predicted_probabilities
        ]
        return batch

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir):
        with open(
            os.path.join(artifacts_dir, "class_to_label.json"), "r", encoding="utf-8"
        ) as fp:
            class_to_label = json.load(fp)
        preprocessor = Preprocessor(class_to_label=class_to_label)
        model = ClassificationModel.load(
            args_fp=os.path.join(artifacts_dir, "args.json"),
            state_dict_fp=os.path.join(artifacts_dir, "model.pt"),
        )
        return cls(preprocessor=preprocessor, model=model)


def batch_metric(batch):
    mcm = multilabel_confusion_matrix(batch["label"], batch["prediction"])
    tn, fp, fn, tp = [], [], [], []
    for i in range(mcm.shape[0]):
        tn.append(mcm[i, 0, 0])
        fp.append(mcm[i, 0, 1])
        fn.append(mcm[i, 1, 0])
        tp.append(mcm[i, 1, 1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ray.init()

    # === Preprocess ===
    print("📦 Preprocessing datasets ...")
    s3 = pyarrow.fs.S3FileSystem(
        anonymous=True,
        region="us-west-2",
        connect_timeout=30,
        request_timeout=60,
        retry_strategy=pyarrow.fs.AwsStandardS3RetryStrategy(max_attempts=10),
    )
    train_ds = ray.data.read_images(
        "s3://doggos-dataset/train", include_paths=True, shuffle="files",
        filesystem=s3,
    )
    train_ds = train_ds.map(add_class)
    val_ds = ray.data.read_images(
        "s3://doggos-dataset/val", include_paths=True, filesystem=s3,
    )
    val_ds = val_ds.map(add_class)

    preprocessor = Preprocessor()
    preprocessor = preprocessor.fit(train_ds, column="class")
    train_ds = preprocessor.transform(ds=train_ds)
    val_ds = preprocessor.transform(ds=val_ds)

    # Cache preprocessed data in Ray object store (no shared filesystem needed).
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()
    gc.collect()  # release preprocessing actors so GPUs are free for training
    print("✅ Preprocessed data cached in Ray object store")

    # === Train ===
    print("🚀 Starting distributed training ...")
    model_store = _ModelStore.options(name="model_store").remote()

    train_loop_config = TRAIN_LOOP_CONFIG.copy()
    train_loop_config["class_to_label"] = preprocessor.class_to_label
    train_loop_config["num_classes"] = len(preprocessor.class_to_label)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=SCALING_CONFIG,
        datasets={"train": train_ds, "val": val_ds},
    )
    results = trainer.fit()
    print("✅ Training complete:", results)

    # === Evaluate ===
    print("📊 Evaluating on test set ...")
    best_artifacts = ray.get(model_store.get.remote())
    model = ClassificationModel(**best_artifacts["args"])
    model.load_state_dict(best_artifacts["state_dict"])
    predictor = TorchPredictor(
        preprocessor=Preprocessor(class_to_label=best_artifacts["class_to_label"]),
        model=model,
    )
    test_ds = ray.data.read_images(
        "s3://doggos-dataset/test", include_paths=True, filesystem=s3,
    )
    test_ds = test_ds.map(add_class)
    test_ds = predictor.preprocessor.transform(ds=test_ds)

    # y_pred (batch inference)
    pred_ds = test_ds.map_batches(predictor, concurrency=4, batch_size=64, num_gpus=1)

    # Aggregated metrics after processing all batches
    metrics_ds = pred_ds.map_batches(batch_metric)
    aggregate_metrics = metrics_ds.sum(["TN", "FP", "FN", "TP"])

    # Aggregate the confusion matrix components across all batches
    tn = aggregate_metrics["sum(TN)"]
    fp = aggregate_metrics["sum(FP)"]
    fn = aggregate_metrics["sum(FN)"]
    tp = aggregate_metrics["sum(TP)"]

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("✅ Evaluation complete:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
