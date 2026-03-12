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

# Clean up existing RayJob
kubectl -n $NAMESPACE delete configmap llm-distributed-inferencing-scripts --ignore-not-found
kubectl -n $NAMESPACE delete rayjob llm-distributed-inferencing --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap llm-distributed-inferencing-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply the kustomize overlay (RayJob)
kubectl apply -k "$OVERLAY_DIR"

# Watch job status
kubectl -n $NAMESPACE get rayjob llm-distributed-inferencing -w

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=llm-distributed-inferencing --tail=200
