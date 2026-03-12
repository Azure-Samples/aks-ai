#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Applying node classes, node pools, and sample apps..."
kubectl apply -f azure/
kubectl apply -f nebius/

echo "⏳ Waiting for all sample app deployments to become ready..."

kubectl rollout status deployment/azure-sample-cpu-app --timeout=3600s
kubectl rollout status deployment/azure-sample-gpu-app --timeout=3600s
kubectl rollout status deployment/nebius-sample-cpu-app --timeout=3600s
kubectl rollout status deployment/nebius-sample-gpu-app --timeout=3600s

echo "✅ All deployments are ready!"

echo "📋 === Azure GPU Deployment Logs ==="
kubectl logs deployment/azure-sample-gpu-app --tail=20

echo "📋 === Nebius GPU Deployment Logs ==="
kubectl logs deployment/nebius-sample-gpu-app --tail=20

echo "🧹 Cleaning up sample app deployments..."
kubectl delete -f azure/cpu_deployment.yaml
kubectl delete -f azure/gpu_deployment.yaml
kubectl delete -f nebius/cpu_deployment.yaml
kubectl delete -f nebius/gpu_deployment.yaml

echo "⏳ Waiting for all nodeclaims to be deleted..."
kubectl wait --for=delete nodeclaim --all --timeout=3600s

echo "🗑️ All sample app deployments and nodeclaims cleaned up."