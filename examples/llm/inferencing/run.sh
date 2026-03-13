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
kubectl -n $NAMESPACE delete configmap llm-inferencing-scripts --ignore-not-found
kubectl -n $NAMESPACE delete rayjob llm-inferencing --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap llm-inferencing-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply the kustomize overlay (RayJob)
kubectl apply -k "$OVERLAY_DIR"

# Wait for RayCluster to be ready
echo "Waiting for RayCluster to be ready..."
until CLUSTER_NAME=$(kubectl -n $NAMESPACE get raycluster -l ray.io/originated-from-cr-name=llm-inferencing -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && [[ -n "$CLUSTER_NAME" ]]; do
    sleep 2
done
kubectl -n $NAMESPACE wait --for=condition=RayClusterProvisioned raycluster/"$CLUSTER_NAME" --timeout=600s
echo "RayCluster $CLUSTER_NAME is ready."

# Wait for RayJob to be deployed (job submission)
echo "Waiting for RayJob to deploy..."
kubectl -n $NAMESPACE wait --for=condition=Complete rayjob/llm-inferencing --timeout=600s &
WAIT_PID=$!

# Stream logs once the job pod appears
echo "Waiting for job pod to start..."
while ! kubectl -n $NAMESPACE get pods -l job-name=llm-inferencing --field-selector=status.phase=Running -o name 2>/dev/null | grep -q .; do
    if ! kill -0 $WAIT_PID 2>/dev/null; then
        # Job already completed before we could stream
        echo "Job finished before logs could be streamed."
        kubectl -n $NAMESPACE logs -l job-name=llm-inferencing --tail=200
        exit 0
    fi
    sleep 2
done

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=llm-inferencing --tail=200

# Wait for job completion
wait $WAIT_PID
echo "RayJob completed."
