from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from optimum.gptq import GPTQQuantizer
import torch
import os 
import json

# Set up access token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "PeterAM4/EPFL-TA-Meister"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

#Need to change this to our own dataset
# Dataset id from Hugging Face 
dataset_id = "wikitext2"

# GPTQ quantizer
quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=1600)
quantizer.quant_method = "gptq"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False) # bug with fast tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16) # we load the model in fp16 on purpose
tokenizer.pad_token = tokenizer.eos_token

# quantize the model 
quantized_model = quantizer.quantize_model(model, tokenizer)

# save the quantize model to disk
save_folder = "checkpoints/quantized_llama_1"
quantized_model.save_pretrained(save_folder, safe_serialization=True)

# load fresh, fast tokenizer and save it to disk
tokenizer = AutoTokenizer.from_pretrained(model_id).save_pretrained(save_folder)

# save quantize_config.json for TGI 
with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
  quantizer.disable_exllama = False
  json.dump(quantizer.to_dict(), f, indent=2)

with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
  config = json.load(f)
  config["quantization_config"]["disable_exllama"] = False
  with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)