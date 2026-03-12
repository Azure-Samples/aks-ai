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

# Install kuberay (skip if already deployed)
if ! helm status kuberay-operator -n $NAMESPACE &>/dev/null; then
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    helm install kuberay-operator kuberay/kuberay-operator \
        --version 1.5.1 \
        --create-namespace \
        --namespace $NAMESPACE \
        --set nodeSelector.agentpool=system \
        --wait
fi

kubectl -n $NAMESPACE delete configmap multimodel-distributed-training-scripts --ignore-not-found
kubectl -n $NAMESPACE delete rayjob multimodel-distributed-training --ignore-not-found

# Create the ConfigMap holding the job script
kubectl create configmap multimodel-distributed-training-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply the kustomize overlay (RayJob)
kubectl apply -k "$OVERLAY_DIR"

# Wait for the job's pod to be running before streaming logs
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE  --timeout=3600s
kubectl wait --for=condition=Ready pod -l job-name=multimodel-distributed-training -n $NAMESPACE --timeout=3600s

kubectl -n $NAMESPACE logs -f -l job-name=multimodel-distributed-training --tail=100
