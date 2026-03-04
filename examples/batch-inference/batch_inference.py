"""
Batch Inference on AKS with KubeRay

Adapted from: https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/01-Batch-Inference.html

This script runs on the RayCluster (submitted via RayJob). It:
  1. Reads dog breed images from a public S3 bucket (distributed CPU read)
  2. Adds class labels extracted from file paths (distributed CPU map)
  3. Generates CLIP embeddings in batches (distributed GPU map_batches)
  4. Materializes embeddings into Ray's shared memory object store
  5. Demonstrates similarity search on the generated embeddings
"""

import os
import ray

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from io import BytesIO
import requests
from doggos.embed import get_top_matches, display_top_matches


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "openai/clip-vit-base-patch32"
S3_DATASET_PATH = "s3://doggos-dataset/train"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_GPU_ACTORS = int(os.environ.get("NUM_GPU_ACTORS", "4"))
SAMPLE_IMAGE_URL = os.environ.get(
    "SAMPLE_IMAGE_URL",
    "https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png",
)
TOP_K = int(os.environ.get("TOP_K", "5"))


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def add_class(row):
    """Extract the breed class from the file path."""
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


# ---------------------------------------------------------------------------
# Embedding actor  — runs on GPU workers via Ray Data map_batches
# ---------------------------------------------------------------------------
class EmbedImages(object):
    """Stateful actor that loads CLIP once and embeds image batches."""

    def __init__(self, model_id: str, device: str):
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.device = device

    def __call__(self, batch):
        # Load and preprocess images
        images = [
            Image.fromarray(np.uint8(img)).convert("RGB") for img in batch["image"]
        ]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            self.device
        )

        # Generate embeddings
        with torch.inference_mode():
            batch["embedding"] = self.model.get_image_features(**inputs).cpu().numpy()

        return batch


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ray.init()

    # ── 1. Data ingestion (distributed CPU read) ────────────────────────
    print(f"Reading images from {S3_DATASET_PATH} ...")
    ds = ray.data.read_images(
        S3_DATASET_PATH,
        include_paths=True,
        shuffle="files",
    )

    # ── 2. Add class labels (distributed CPU map) ───────────────────────
    ds = ds.map(add_class)

    # ── 3. Batch embedding generation (distributed GPU map_batches) ─────
    print(
        f"Generating CLIP embeddings with {NUM_GPU_ACTORS} GPU actors, "
        f"batch_size={BATCH_SIZE} ..."
    )
    embeddings_ds = ds.map_batches(
        EmbedImages,
        fn_constructor_kwargs={
            "model_id": MODEL_ID,
            "device": "cuda",
        },
        concurrency=NUM_GPU_ACTORS,
        batch_size=BATCH_SIZE,
        num_gpus=1,
    )
    embeddings_ds = embeddings_ds.drop_columns(["image"])  # remove image column

    # ── 4. Materialize into Ray object store ─────────────────────────────
    print("Materializing embeddings into Ray object store ...")
    embeddings_ds = embeddings_ds.materialize()
    print(f"Embeddings materialized — {embeddings_ds.count()} rows in object store.")

    # ── 5. Similarity search demo ───────────────────────────────────────
    print(f"\nRunning similarity search against: {SAMPLE_IMAGE_URL}")

    resp = requests.get(SAMPLE_IMAGE_URL, timeout=30)
    resp.raise_for_status()
    query_image = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))

    embed_fn = EmbedImages(model_id=MODEL_ID, device="cpu")
    query_embedding = embed_fn({"image": [query_image]})["embedding"][0]
    print(f"Query embedding shape: {np.shape(query_embedding)}")

    # Use the materialized dataset directly from the object store
    top_matches = get_top_matches(query_embedding, embeddings_ds, n=TOP_K)
    display_top_matches(SAMPLE_IMAGE_URL, top_matches)

    print(f"\nTop {TOP_K} similar images:")
    for i, match in enumerate(top_matches, 1):
        print(
            f"  {i}. class={match['class']:<20s}  "
            f"similarity={match['similarity']:.4f}  "
            f"path={match['path']}"
        )

    print("\nBatch inference pipeline completed.")


if __name__ == "__main__":
    main()
