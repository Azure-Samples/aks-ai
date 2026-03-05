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


## Inference Benchmark (RayJob on KubeRay)

NAMESPACE=ray

# Clean up existing RayJob
kubectl -n $NAMESPACE delete rayjob llm-inferencing-benchmark --ignore-not-found

# Create the ConfigMap from the actual script file
kubectl create configmap inference-benchmark-scripts \
    --from-file=examples/llm-inferencing-ray/benchmark.py \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the RayJob
kubectl apply -f examples/llm-inferencing-ray/rayjob.yaml

# Watch job status
kubectl -n $NAMESPACE get rayjob llm-inferencing-benchmark -w

# Stream logs
kubectl -n $NAMESPACE logs -f -l job-name=llm-inferencing-benchmark --tail=200
