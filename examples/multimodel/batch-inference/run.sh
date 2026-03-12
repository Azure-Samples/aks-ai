#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

### Inference Benchmark

NAMESPACE=ray

# Clean up existing job
kubectl -n $NAMESPACE delete rayjob multimodel-batch-inference --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap multimodel-batch-inference-scripts \
    --from-file="$SCRIPT_DIR/batch_inference.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the Job
kubectl apply -f "$SCRIPT_DIR/rayjob.yaml"

# Wait for the pod to be running
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE  --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=multimodel-batch-inference -n $NAMESPACE --timeout=600s
kubectl -n $NAMESPACE get pods -o wide

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=multimodel-batch-inference --tail=200
