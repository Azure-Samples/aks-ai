#!/bin/bash

set -eo pipefail

aks-flex-cli config env --nebius > .env

aks-flex-cli network deploy

aks-flex-cli aks deploy --unbounded-cni --dra-driver

aks-flex-cli config unbounded-cni site \
  --name site-remote \
  --node-cidr 172.20.0.0/16 \
  --pod-cidr 10.200.0.0/16 \
  | kubectl apply -f -

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
  --set "serviceAccount.annotations.azure\.workload\.identity/client-id=<karpenter-flex-client-id>" \
  --set-string "podLabels.azure\.workload\.identity/use=true" \
  --set "controller.env[0].name=ARM_CLOUD,controller.env[0].value=AzurePublicCloud" \
  --set "controller.env[1].name=LOCATION,controller.env[1].value=southcentralus" \
  --set "controller.env[2].name=ARM_RESOURCE_GROUP,controller.env[2].value=rg-aks-flex-<username>" \
  --set "controller.env[3].name=AZURE_TENANT_ID,controller.env[3].value=<tenant-id>" \
  --set "controller.env[4].name=AZURE_CLIENT_ID,controller.env[4].value=<karpenter-flex-client-id>" \
  --set "controller.env[5].name=AZURE_SUBSCRIPTION_ID,controller.env[5].value=<subscription-id>" \
  --set "controller.env[6].name=AZURE_NODE_RESOURCE_GROUP,controller.env[6].value=<node-resource-group>" \
  --set "controller.env[7].name=VNET_SUBNET_ID,controller.env[7].value=/subscriptions/.../subnets/aks" \
  --set "controller.env[8].name=KUBELET_BOOTSTRAP_TOKEN,controller.env[8].value=<token-id>.<token-secret>" \
  --set-string "controller.env[9].name=DISABLE_LEADER_ELECTION,controller.env[9].value=false"

kubectl create secret generic nebius-credentials \
  --namespace karpenter \
  --from-file="credentials.json=/home/ansonqian/.nebius/serviceaccount-e00y4ya0mjry9dbtc0-credentials.json" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f configs/nebius/