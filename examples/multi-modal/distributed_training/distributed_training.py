"""
Distributed Training on AKS with KubeRay

Adapted from: https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/02-Distributed-Training.html

This script runs on the RayCluster (submitted via RayJob). It:
  1. Preprocesses dog breed images using CLIP embeddings (Ray Data + GPU actors)
  2. Trains a classifier with Ray Train TorchTrainer (distributed DDP)
  3. Logs metrics/artifacts to MLflow on shared PVC storage
  4. Evaluates the model on a held-out test set
"""

import os
import ray

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from doggos.embed import EmbedImages

import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import ray.train
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import multilabel_confusion_matrix


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STORAGE_PATH = "/mnt/cluster_storage"
PREPROCESSED_DATA_PATH = os.path.join(STORAGE_PATH, "doggos/preprocessed_data")
MODEL_REGISTRY = os.path.join(STORAGE_PATH, "mlflow/doggos")

EXPERIMENT_NAME = "doggos"
TRAIN_LOOP_CONFIG = {
    "model_registry": MODEL_REGISTRY,
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
NUM_WORKERS = 4
SCALING_CONFIG = ray.train.ScalingConfig(
    num_workers=NUM_WORKERS,
    use_gpu=True,
    resources_per_worker={"CPU": 8, "GPU": 2},
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


class Preprocessor:
    """Preprocessor class."""

    def __init__(self, class_to_label=None):
        self.class_to_label = class_to_label or {}
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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_classes = num_classes

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
        return torch.argmax(z, dim=1).cpu().numpy()

    @torch.inference_mode()
    def predict_probabilities(self, batch):
        z = self(batch)
        return F.softmax(z, dim=1).cpu().numpy()

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
    from ray.train.torch import get_device

    dtypes = {"embedding": torch.float32, "label": torch.int64}
    tensor_batch = {}
    if device is None:
        try:
            device = get_device()
        except RuntimeError:
            device = "cpu"
    for key in dtypes:
        if key in batch:
            tensor_batch[key] = torch.as_tensor(
                batch[key], dtype=dtypes[key], device=device
            )
    return tensor_batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(ds, batch_size, model, num_classes, loss_fn, optimizer):
    model.train()
    loss = 0.0
    for i, batch in enumerate(
        ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    ):
        optimizer.zero_grad()
        z = model(batch)
        targets = F.one_hot(batch["label"], num_classes=num_classes).float()
        J = loss_fn(z, targets)
        J.backward()
        optimizer.step()
        loss += (J.detach().item() - loss) / (i + 1)
    return loss


def eval_epoch(ds, batch_size, model, num_classes, loss_fn):
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(
            ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
        ):
            z = model(batch)
            targets = F.one_hot(batch["label"], num_classes=num_classes).float()
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["label"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config):
    model_registry = config["model_registry"]
    experiment_name = config["experiment_name"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # MLflow tracking (rank 0 only).
    if ray.train.get_context().get_world_rank() == 0:
        mlflow.set_tracking_uri(f"file:{model_registry}")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(config)

    # Datasets.
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    # Model.
    model = ClassificationModel(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_p=config["dropout_p"],
        num_classes=num_classes,
    )
    model = ray.train.torch.prepare_model(model)

    # Optimizer & scheduler.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
    )

    # Train.
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            train_ds, batch_size, model, num_classes, loss_fn, optimizer
        )
        val_loss, _, _ = eval_epoch(val_ds, batch_size, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        with tempfile.TemporaryDirectory() as dp:
            model.module.save(dp=dp)
            metrics = dict(
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                val_loss=val_loss,
            )
            with open(os.path.join(dp, "class_to_label.json"), "w") as fp:
                json.dump(config["class_to_label"], fp, indent=4)
            if ray.train.get_context().get_world_rank() == 0:
                mlflow.log_metrics(metrics, step=epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.log_artifacts(dp)

    if ray.train.get_context().get_world_rank() == 0:
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Evaluation helpers
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

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir):
        with open(os.path.join(artifacts_dir, "class_to_label.json"), "r") as fp:
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
    train_ds = ray.data.read_images(
        "s3://doggos-dataset/train", include_paths=True, shuffle="files"
    )
    train_ds = train_ds.map(add_class)
    val_ds = ray.data.read_images("s3://doggos-dataset/val", include_paths=True)
    val_ds = val_ds.map(add_class)

    preprocessor = Preprocessor()
    preprocessor = preprocessor.fit(train_ds, column="class")
    train_ds = preprocessor.transform(ds=train_ds)
    val_ds = preprocessor.transform(ds=val_ds)

    # Write preprocessed data to shared PVC storage.
    if os.path.exists(PREPROCESSED_DATA_PATH):
        shutil.rmtree(PREPROCESSED_DATA_PATH)
    preprocessed_train_path = os.path.join(PREPROCESSED_DATA_PATH, "preprocessed_train")
    preprocessed_val_path = os.path.join(PREPROCESSED_DATA_PATH, "preprocessed_val")
    train_ds.write_parquet(preprocessed_train_path)
    val_ds.write_parquet(preprocessed_val_path)
    print("✅ Preprocessed data written to", PREPROCESSED_DATA_PATH)

    # === Train ===
    print("🚀 Starting distributed training ...")
    preprocessed_train_ds = ray.data.read_parquet(preprocessed_train_path)
    preprocessed_val_ds = ray.data.read_parquet(preprocessed_val_path)

    if os.path.isdir(MODEL_REGISTRY):
        shutil.rmtree(MODEL_REGISTRY)
    os.makedirs(MODEL_REGISTRY, exist_ok=True)

    train_loop_config = TRAIN_LOOP_CONFIG.copy()
    train_loop_config["class_to_label"] = preprocessor.class_to_label
    train_loop_config["num_classes"] = len(preprocessor.class_to_label)

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=SCALING_CONFIG,
        datasets={"train": preprocessed_train_ds, "val": preprocessed_val_ds},
    )
    results = trainer.fit()
    print("✅ Training complete:", results)

    # === Evaluate ===
    print("📊 Evaluating on test set ...")
    mlflow.set_tracking_uri(f"file:{MODEL_REGISTRY}")
    sorted_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.val_loss ASC"],
    )
    best_run = sorted_runs.iloc[0]
    artifacts_dir = urlparse(best_run.artifact_uri).path

    predictor = TorchPredictor.from_artifacts_dir(artifacts_dir=artifacts_dir)
    test_ds = ray.data.read_images("s3://doggos-dataset/test", include_paths=True)
    test_ds = test_ds.map(add_class)
    test_ds = predictor.preprocessor.transform(ds=test_ds)

    pred_ds = test_ds.map_batches(predictor, concurrency=4, batch_size=64, num_gpus=1)
    metrics_ds = pred_ds.map_batches(batch_metric)
    agg = metrics_ds.sum(["TN", "FP", "FN", "TP"])

    tn, fp, fn, tp = agg["sum(TN)"], agg["sum(FP)"], agg["sum(FN)"], agg["sum(TP)"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"✅ Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")


if __name__ == "__main__":
    main()
