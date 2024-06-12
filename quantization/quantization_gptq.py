import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset, concatenate_datasets
import torch
from data_processing import load_and_process_data

# HF token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "PeterAM4/EPFL-TA-Meister"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

text_data = load_and_process_data()

# Define GPTQ configuration with optimal parameters
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    damp_percent=0.1,
    dataset=text_data,
    tokenizer=tokenizer,
    sequence_length=2048  # Optional, specify if known chosen to prevent OOM
)

# Load and quantize model
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "cpu"},  # Force loading on CPU, quantization will still be on GPU justas much on CPU as possible
    max_memory={0: "0GiB", "cpu": "80GiB"},  # Allocate memory for CPU
    quantization_config=gptq_config
)

# Push the quantized model to the Hub
quantized_model.push_to_hub("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")
tokenizer.push_to_hub("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")
