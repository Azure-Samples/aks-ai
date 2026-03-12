#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

### Inference Benchmark (RayJob on KubeRay)

NAMESPACE=ray

# Clean up existing RayJob
kubectl -n $NAMESPACE delete rayjob llm-inferencing-benchmark --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap inference-benchmark-scripts \
    --from-file="$SCRIPT_DIR/benchmark.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the RayJob
kubectl apply -f "$SCRIPT_DIR/rayjob.yaml"

# Watch job status
kubectl -n $NAMESPACE get rayjob llm-inferencing-benchmark -w

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=llm-inferencing-benchmark --tail=200
