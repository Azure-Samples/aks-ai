# KubeRay: Entity Recognition E2E

## Entity Recognition E2E (Training + Batch Inference)

> **Note:** No shared PVC is required. The script uses Ray object store to
> distribute dataset and checkpoint files between head and GPU worker nodes.

### 1. Create the ConfigMap holding the job script

```bash
NAMESPACE=ray
kubectl create configmap fine-tune-scripts \
  --from-file=fine_tune.py \
  -n $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Submit the RayJob (creates its own transient cluster)

```bash
kubectl apply -f rayjob.yaml
```

### 3. Wait for the job's pod to be running before streaming logs

```bash
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=llm -n $NAMESPACE --timeout=300s
kubectl -n $NAMESPACE get pods -o wide
```

### 4. Watch progress

```bash
kubectl -n $NAMESPACE logs -f -l job-name=llm --tail=100
```

### 5. Clean up after inspection

```bash
kubectl -n $NAMESPACE delete rayjob llm
```

> **Note:** Replace `$NAMESPACE` with your target Kubernetes namespace.
