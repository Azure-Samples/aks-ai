### LLM Fine-Tuning (Entity Recognition)

NAMESPACE=ray

# Install kuberay (skip if already deployed)
if ! helm status kuberay-operator -n $NAMESPACE &>/dev/null; then
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    helm install kuberay-operator kuberay/kuberay-operator \
        --version 1.5.1 \
        --create-namespace \
        --namespace $NAMESPACE \
        --set nodeSelector.agentpool=system \
        --wait
fi

# Clean up existing rayjob
kubectl -n $NAMESPACE delete rayjob distributed-finetuning --ignore-not-found

# Create the ConfigMap holding the job script
kubectl create configmap fine-tune-scripts \
    --from-file=examples/llm-fine-tuning/fine_tune.py \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Submit the RayJob (creates its own transient cluster)
kubectl apply -f examples/llm-fine-tuning/rayjob.yaml

# Wait for the job's pod to be running before streaming logs
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/created-by=kuberay-operator -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=Ready pod -l job-name=distributed-finetuning -n $NAMESPACE --timeout=300s
kubectl -n $NAMESPACE get pods -o wide

kubectl -n $NAMESPACE logs -f -l job-name=distributed-finetuning --tail=100
