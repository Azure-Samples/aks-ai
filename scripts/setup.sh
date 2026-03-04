#!/bin/bash

set -eo pipefail

aks-flex-cli config env --nebius > .env

aks-flex-cli network deploy

aks-flex-cli aks deploy --cilium --wireguard --gpu-device-plugin

aks-flex-cli config karpenter helm

helm upgrade --install karpenter charts/karpenter \
  --namespace karpenter \
  --create-namespace \
  --set settings.clusterName="aks" \
  --set settings.clusterEndpoint="https://aks-xxxx.hcp.eastus2.azmk8s.io:443" \
  --set logLevel=debug \
  --set replicas=1 \
  --set controller.nebiusCredentials.enabled=true \
  --set controller.image.digest="" \
  --set "controller.env[0].name=ARM_CLOUD,controller.env[0].value=AzurePublicCloud" \
  --set "controller.env[1].name=LOCATION,controller.env[1].value=southcentralus" \
  --set "controller.env[2].name=ARM_RESOURCE_GROUP,controller.env[2].value=rg-aks-flex-<username>" \
  --set "controller.env[3].name=AZURE_TENANT_ID,controller.env[3].value=<tenant-id>" \
  --set "controller.env[4].name=AZURE_SUBSCRIPTION_ID,controller.env[4].value=<subscription-id>" \
  --set "controller.env[5].name=AZURE_NODE_RESOURCE_GROUP,controller.env[5].value=<node-resource-group>" \
  --set "controller.env[6].name=SSH_PUBLIC_KEY,controller.env[6].value=ssh-ed25519 AAAA..." \
  --set "controller.env[7].name=VNET_SUBNET_ID,controller.env[7].value=/subscriptions/.../subnets/nodes" \
  --set "controller.env[8].name=KUBELET_BOOTSTRAP_TOKEN,controller.env[8].value=<token-id>.<token-secret>" \
  --set-string "controller.env[9].name=DISABLE_LEADER_ELECTION,controller.env[9].value=false"

kubectl create secret generic nebius-credentials \
  --namespace karpenter \
  --from-file="credentials.json=<path-to-credentials-file>" \
  --dry-run=client -o yaml | kubectl apply -f -

kubect apply -f configs/nebius/