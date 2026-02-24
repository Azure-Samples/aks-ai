import json
import textwrap
from IPython.display import Code, Image, display
import os
from pathlib import Path
import yaml

with open("cluster_storage/viggo/train.jsonl", "r") as fp:
    first_line = fp.readline()
    item = json.loads(first_line)
system_content = item["instruction"]
print(textwrap.fill(system_content, width=80))

display(Code(filename="cluster_storage/viggo/dataset_info.json", language="json"))
display(Code(filename="llama3_lora_sft_ray.yaml", language="yaml"))

model_id = "ft-model"  # call it whatever you want
model_source = yaml.safe_load(open("llama3_lora_sft_ray.yaml"))["model_name_or_path"]  # HF model ID, S3 mirror config, or GCS mirror config
print (model_source)