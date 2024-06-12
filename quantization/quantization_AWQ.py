import os
import re
import sys
import time
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
from data_processing import load_and_process_data

access_token = "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"
os.environ["HF_TOKEN"] = access_token

# Set up the model ID and Hugging Face token
model_path = "PeterAM4/EPFL-TA-Meister"
text_data = load_and_process_data()

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=text_data)

quant_path = "mimimamalah/EPFL-TA-Meister-AWQ-4bit"
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)