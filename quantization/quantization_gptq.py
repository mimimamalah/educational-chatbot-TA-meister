import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset, concatenate_datasets
import torch

# HF token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "PeterAM4/EPFL-TA-Meister"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")


# Transformation functions for datasets
def transform_example(example):
    return {"text": f"Question: {example['problem']} Answer: {example['solution']}"}

def transform_physics_example(example):
    return {"text": f"Message: {example['message_1']} Reply: {example['message_2']}"}

def transform_code_example(example):
    return {"text": f"Query: {example['query']} Answer: {example['answer']}"}

def transform_prompt_completion(example):
    return {"text": f"{example['prompt']} {example['completion']}"}

# Load and transform datasets
train_dataset = load_dataset("json", data_files="./data/sft_train_m1.json")['train'].map(transform_prompt_completion)
new_train_dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True).map(transform_example)
new_test_dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True).map(transform_example)
stackexchange_dataset = load_dataset("json", data_files="./data/sft_stackexchange_43043.json", split="train").map(transform_prompt_completion)
physics_dataset = load_dataset("lgaalves/camel-ai-physics", split="train").shuffle(seed=42).select(range(10000)).map(transform_physics_example)
code_feedback_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train").shuffle(seed=42).select(range(10000)).map(transform_code_example)
wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train") # subset of wikitext

print(wikitext_dataset)

# Combine and shuffle datasets
combined_train_dataset = concatenate_datasets([
    train_dataset,
    new_train_dataset,
    new_test_dataset,
    stackexchange_dataset,
    physics_dataset,
    code_feedback_dataset,
    wikitext_dataset
]).shuffle(seed=42)

# Filter the combined dataset to include only the 'text' field
combined_train_dataset = combined_train_dataset.select_columns(["text"])

print(combined_train_dataset)

# Convert combined dataset to list of strings for GPTQ quantization
text_data = [item['text'] for item in combined_train_dataset.select(range(8000))] # paper uses 128 we used more to keep as much as possible performance on the datasets

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