VERSION=1.5.1
NAMESPACE=ray
helm install raycluster kuberay/ray-cluster --create-namespace -n $NAMESPACE --atomic \
  --version $VERSION -f ./raycluster.yaml

kubectl create configmap llm-job-sample --from-file=llm_job.py -n $NAMESPACE

helm uninstall raycluster -n $NAMESPACE

### Entity Recognition E2E  (training + batch inference)
# 1. Create the ConfigMap holding the job script
kubectl create configmap entity-recognition-scripts \
  --from-file=entity_recognition.py \
  -n $NAMESPACE

# 2. Submit the RayJob (creates its own transient cluster)
kubectl apply -f rayjob-entity-recognition.yaml

# 3. Watch progress
kubectl -n $NAMESPACE logs -f -l ray.io/cluster=entity-recognition --tail=200

# 4. Clean up after inspection
kubectl delete rayjob entity-recognition -n $NAMESPACE

### Submit to an *existing* RayCluster instead of a standalone RayJob
# kubectl -n $NAMESPACE port-forward svc/raycluster-kuberay-head-svc 8265:8265 &
# ray job submit --address http://localhost:8265 \
#   --runtime-env-json='{"pip":["pyyaml","xgrammar==0.1.11","pynvml==12.0.0","hf_transfer==0.1.9","tensorboard==2.19.0","llamafactory@git+https://github.com/hiyouga/LLaMA-Factory.git@v0.9.4#egg=llamafactory","vllm>=0.8.0"]}' \
#   --working-dir . \
#   -- python entity_recognition.py

### Online serving (after training finishes)
# kubectl -n $NAMESPACE port-forward svc/raycluster-kuberay-head-svc 8265:8265 &
# ray job submit --address http://localhost:8265 \
#   --runtime-env-json='{"pip":["vllm>=0.8.0","pyyaml"]}' \
#   --no-wait \
#   -- python serve_model.py --lora-path /mnt/cluster_storage/viggo/saves/lora_sft_ray/<LATEST_TRAINER_DIR>/checkpoint_000000/checkpoint

### LLM

rm -rf cluster_storage/viggo  # clean up
mkdir -p cluster_storage/viggo
wget https://viggo-ds.s3.amazonaws.com/train.jsonl -O cluster_storage/viggo/train.jsonl
wget https://viggo-ds.s3.amazonaws.com/val.jsonl -O cluster_storage/viggo/val.jsonl
wget https://viggo-ds.s3.amazonaws.com/test.jsonl -O cluster_storage/viggo/test.jsonl
wget https://viggo-ds.s3.amazonaws.com/dataset_info.json -O cluster_storage/viggo/dataset_info.json

head -n 1 cluster_storage/viggo/train.jsonl | python3 -m json.tool