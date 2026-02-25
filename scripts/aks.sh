#!/bin/bash

set -eo pipefail

source scripts/variables.sh

az account set -s ${SUBSCRIPTION}
if az group show -n ${RESOURCE_GROUP} &>/dev/null; then
    echo "Resource group already exists."
else
    echo "Resource group does not exist. Creating ..."
    az group create -l ${LOCATION} -n ${RESOURCE_GROUP}
fi

if az aks show -g ${RESOURCE_GROUP} -n ${CLUSTER_NAME} &>/dev/null; then
    echo "Cluster already exists."
else
    echo "Cluster does not exist. Creating ..."
    MY_USER_ID=$(az ad signed-in-user show --query id -o tsv)
    az aks create -l "${LOCATION}" \
        -g "${RESOURCE_GROUP}" \
        -n "${CLUSTER_NAME}" \
        --tier standard \
        --kubernetes-version 1.34.2 \
        --disable-disk-driver \
        --disable-file-driver \
        --enable-blob-driver \
        --enable-aad \
        --aad-admin-group-object-ids "$MY_USER_ID" \
        --nodepool-name ${SYSTEM_POOL_NAME} \
        --node-vm-size ${SYSTEM_VM_SIZE} \
        --node-count ${SYSTEM_POOL_SIZE} \
        --network-plugin azure
        # --network-plugin none
fi


if az aks nodepool show --resource-group ${RESOURCE_GROUP} --cluster-name ${CLUSTER_NAME} --name user &>/dev/null; then
    echo "User pool already exists."
else
    echo "User pool does not exist. Creating ..."
     az aks nodepool add \
        --resource-group ${RESOURCE_GROUP} \
        --cluster-name ${CLUSTER_NAME} \
        --node-vm-size ${USER_VM_SIZE} \
        --node-count ${USER_POOL_SIZE} \
        --name user
fi

az aks get-credentials --resource-group ${RESOURCE_GROUP} \
    --name ${CLUSTER_NAME} \
    --admin \
    --overwrite-existing
