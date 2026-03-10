### Ray Benchmark: Ray head/worker interaction

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE=default

echo "=== Deploying ray benchmark workload ==="

# Clean up any previous run
kubectl delete deployment ray-benchmark-head ray-benchmark-worker --ignore-not-found
kubectl delete svc ray-head-svc --ignore-not-found
kubectl delete configmap ray-benchmark-scripts --ignore-not-found

# Create ConfigMap from the actual Python scripts
kubectl create configmap ray-benchmark-scripts \
    --from-file="$SCRIPT_DIR/examples/ray-benchmark/head_server.py" \
    --from-file="$SCRIPT_DIR/examples/ray-benchmark/worker_client.py" \
    -n $NAMESPACE

# Deploy head service and worker pods
kubectl apply -f "$SCRIPT_DIR/examples/ray-benchmark/manifests.yaml" -n $NAMESPACE

echo "=== Waiting for head pod to be ready ==="
kubectl wait --for=condition=Ready pod -l app=ray-benchmark-head -n $NAMESPACE --timeout=300s

echo "=== Waiting for worker pods to be ready ==="
kubectl wait --for=condition=Ready pod -l app=ray-benchmark-worker -n $NAMESPACE --timeout=600s

echo "=== Streaming worker logs ==="
kubectl logs -f -l app=ray-benchmark-worker -n $NAMESPACE --tail=200
