# Set up pvc with blobfuse2
kubectl apply -f configs/storageclass.yaml
kubectl apply -f configs/pvc.yaml

### Entity Recognition E2E  (training + batch inference)

NAMESPACE=ray

# 1. Create the ConfigMap holding the job script
kubectl create configmap entity-recognition-scripts \
  --from-file=entity_recognition.py \
  -n $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Submit the RayJob (creates its own transient cluster)
kubectl apply -f llm-rayjob.yaml

# 3. Wait for the job's pod to be running before streaming logs
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE  --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=llm -n $NAMESPACE --timeout=300s
kubectl -n $NAMESPACE get pods -o wide

# 4. Watch progress
kubectl -n $NAMESPACE logs -f -l job-name=llm --tail=100

# 5. Clean up after inspection
kubectl -n $NAMESPACE delete rayjob llm

