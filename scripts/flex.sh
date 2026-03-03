#!/bin/bash

set -eo pipefail

aks-flex-cli config env --nebius > .env

aks-flex-cli network deploy

aks-flex-cli aks deploy --cilium --wireguard --gpu-device-plugin
