#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

### Inference Benchmark (RayJob on KubeRay)

NAMESPACE=ray

# Clean up existing RayJob
kubectl -n $NAMESPACE delete rayjob llm-distributed-inferencing --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap llm-distributed-inferencing-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the RayJob
kubectl apply -f "$SCRIPT_DIR/rayjob.yaml"

# Watch job status
kubectl -n $NAMESPACE get rayjob llm-distributed-inferencing -w

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=llm-distributed-inferencing --tail=200
