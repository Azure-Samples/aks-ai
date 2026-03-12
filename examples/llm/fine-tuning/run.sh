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

# Clean up existing rayjob
kubectl -n $NAMESPACE delete configmap llm-fine-tuning-scripts --ignore-not-found
kubectl -n $NAMESPACE delete rayjob llm-fine-tuning --ignore-not-found

# Create the ConfigMap holding the job script
kubectl create configmap llm-fine-tuning-scripts \
    --from-file="$SCRIPT_DIR/main.py" \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply the kustomize overlay (RayJob)
kubectl apply -k "$OVERLAY_DIR"

# Wait for the job's pod to be running before streaming logs
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=llm-fine-tuning -n $NAMESPACE --timeout=300s
kubectl -n $NAMESPACE get pods -o wide

kubectl -n $NAMESPACE logs -f -l job-name=llm-fine-tuning --tail=100
