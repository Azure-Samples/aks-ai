FROM rayproject/ray:2.53.0-py312-gpu

# Install inference dependencies so they are baked into the image
# instead of being downloaded at runtime via runtimeEnvYAML.
RUN pip install --no-cache-dir \
    "ray[llm]==2.53.0" \
    "vllm>=0.12.0,<0.13.0"
