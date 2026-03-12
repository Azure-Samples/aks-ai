#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLOUD=${1:?Usage: $0 <azure|nebius>}
NAMESPACE=ray

if [[ "$CLOUD" != "azure" && "$CLOUD" != "nebius" ]]; then
    echo "Error: CLOUD must be 'azure' or 'nebius', got '$CLOUD'"
    exit 1
fi

OVERLAY_DIR="$SCRIPT_DIR/overlays/$CLOUD"

# Clean up existing job
kubectl -n $NAMESPACE delete configmap multimodel-batch-inference-scripts --ignore-not-found
kubectl -n $NAMESPACE delete rayjob multimodel-batch-inference --ignore-not-found

# Create the ConfigMap holding the job script
kubectl create configmap multimodel-batch-inference-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply the kustomize overlay (RayJob)
kubectl apply -k "$OVERLAY_DIR"

# Wait for the pod to be running
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE  --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=multimodel-batch-inference -n $NAMESPACE --timeout=600s
kubectl -n $NAMESPACE get pods -o wide

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=multimodel-batch-inference --tail=200
