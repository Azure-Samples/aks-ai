### Inference Benchmark

NAMESPACE=default

# Clean up existing job
kubectl -n $NAMESPACE delete job inference-benchmark --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap inference-benchmark-script \
    --from-file=examples/llm-inferencing/benchmark.py \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the Job
kubectl apply -f examples/llm-inferencing/job.yaml

# Wait for the pod to be running
kubectl wait --for=condition=Ready pod -l job-name=inference-benchmark -n $NAMESPACE --timeout=600s
kubectl -n $NAMESPACE get pods -o wide

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=inference-benchmark --tail=200
