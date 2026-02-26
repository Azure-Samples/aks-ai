"""
Online serving of the fine-tuned model with Ray Serve LLM.
===========================================================

Deploys the base model (Qwen/Qwen2.5-7B-Instruct) with the LoRA adapter
as an OpenAI-compatible endpoint using ray.serve.llm.

Usage — submit to a running RayCluster:
  # Port-forward the Ray dashboard
  kubectl -n ray port-forward svc/raycluster-kuberay-head-svc 8265:8265

  # Submit as a Ray job (stays alive until cancelled)
  ray job submit --address http://localhost:8265 \
      --runtime-env-json='{"pip":["vllm>=0.8.0","pyyaml"]}' \
      -- python serve_model.py --lora-path <LORA_CHECKPOINT_PATH>

To query the endpoint once it's running:
  kubectl -n ray port-forward svc/raycluster-kuberay-head-svc 8000:8000

  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "ft-model:<lora_id>",
      "messages": [
        {"role": "system",  "content": "<system_prompt>"},
        {"role": "user",    "content": "Do you have a favorite ESRB content rating?"}
      ],
      "stream": true
    }'
"""

import argparse
import json
import os
import signal
from pathlib import Path

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

MODEL_SOURCE = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = "ft-model"


def main():
    parser = argparse.ArgumentParser(
        description="Serve fine-tuned model via Ray Serve LLM"
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Local path to the LoRA adapter checkpoint directory",
    )
    parser.add_argument("--model-source", default=MODEL_SOURCE)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    # The dynamic_lora_loading_path is the *parent* of the adapter directory.
    # vLLM will load adapters by name from within that parent directory.
    lora_path = Path(args.lora_path).resolve()
    dynamic_lora_path = str(lora_path.parent)
    lora_id = lora_path.name
    print(f"  Model source          : {args.model_source}")
    print(f"  LoRA adapter path     : {lora_path}")
    print(f"  Dynamic LoRA base dir : {dynamic_lora_path}")
    print(f"  LoRA adapter id       : {lora_id}")
    print(f"  Query with model name : {args.model_id}:{lora_id}")

    llm_config = LLMConfig(
        model_loading_config={
            "model_id": args.model_id,
            "model_source": args.model_source,
        },
        lora_config={
            "dynamic_lora_loading_path": dynamic_lora_path,
            "max_num_adapters_per_replica": 16,
        },
        deployment_config={
            "autoscaling_config": {
                "min_replicas": 1,
                "max_replicas": 2,
            }
        },
        engine_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "enable_lora": True,
        },
    )

    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app)

    # Keep the process alive so that Ray Serve stays up
    print("\n  Serving at http://0.0.0.0:8000/v1  (Ctrl+C to stop)")
    signal.pause()


if __name__ == "__main__":
    main()
