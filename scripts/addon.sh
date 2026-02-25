#!/bin/bash

set -eo pipefail

source scripts/variables.sh

# cilium install --version 1.19.1 --set azure.resourceGroup="${RESOURCE_GROUP}"

kubectl label nodes -l agentpool=${USER_POOL_NAME} nvidia.com/gpu.present=true --overwrite

helm upgrade --install nvdp nvdp/nvidia-device-plugin \
    --version=0.18.2 \
    --create-namespace \
    --namespace nvidia \
    -f configs/device-plugin-values.yaml \
    --wait

helm upgrade --install dra-driver nvidia/nvidia-dra-driver-gpu \
    --version=25.12.0 \
    --create-namespace \
    --namespace nvidia \
    -f configs/dra-driver-values.yaml \
    --wait

helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 \
    --create-namespace \
    --namespace ray \
    --wait