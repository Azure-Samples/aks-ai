#!/bin/bash

set -eo pipefail

aks-flex-cli config env --nebius > .env

echo "🌐 Deploying network..."
aks-flex-cli network deploy

echo "☸️ Deploying AKS cluster..."
aks-flex-cli aks deploy --wireguard --cilium --nvidia-device-plugin --nvidia-dra-driver

echo "⚙️ Configuring Karpenter Helm values..."
aks-flex-cli config karpenter helm \
    --nebius-credentials-file ~/.nebius/credentials.json \
    --ssh-public-key-file ~/.ssh/id_ed25519.pub

echo "🚀 Installing Karpenter..."
helm upgrade --install karpenter charts/karpenter \
  --namespace karpenter \
  --create-namespace \
  --values karpenter_values.yaml