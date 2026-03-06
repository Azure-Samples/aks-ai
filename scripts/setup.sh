#!/bin/bash

set -eo pipefail

aks-flex-cli config env --nebius > .env

echo "Deploy network..."
aks-flex-cli network deploy

echo "Deploy cluster..."
aks-flex-cli aks deploy --wireguard --cilium --nvidia-device-plugin --nvidia-dra-driver

aks-flex-cli config karpenter helm \
    --nebius-credentials-file ~/.nebius/credentials.json \
    --ssh-public-key-file ~/.ssh/id_ed25519.pub

echo "Deploy Karpenter..."
helm upgrade --install karpenter charts/karpenter \
  --namespace karpenter \
  --create-namespace \
  --values karpenter_values.yaml

echo "Create node class, node pool and sample app..."
kubectl apply -f configs/azure/
kubectl apply -f configs/nebius/

echo "Waiting for all sample app deployments to be ready..."

kubectl rollout status deployment/azure-sample-cpu-app --timeout=3600s
kubectl rollout status deployment/azure-sample-gpu-app --timeout=3600s
kubectl rollout status deployment/nebius-sample-cpu-app --timeout=3600s
kubectl rollout status deployment/nebius-sample-gpu-app --timeout=3600s

echo "All deployments are ready!"

echo "=== Azure GPU Deployment Logs ==="
kubectl logs deployment/azure-sample-gpu-app --tail=20

echo "=== Nebius GPU Deployment Logs ==="
kubectl logs deployment/nebius-sample-gpu-app --tail=20

echo "Cleaning up sample app deployments..."
kubectl delete -f configs/azure/cpu_deployment.yaml
kubectl delete -f configs/azure/gpu_deployment.yaml
kubectl delete -f configs/nebius/cpu_deployment.yaml
kubectl delete -f configs/nebius/gpu_deployment.yaml

echo "Sample app deployments deleted."