import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
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
)

# Load and quantize model using accelerate for better device management
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically map model across available devices
    quantization_config=gptq_config
)

# In our case we did not have to change any weight put here just in case
# Check for NaNs and infinite weights and replace them
def check_and_fix_weights(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Fixing parameter: {name}")
            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e20, neginf=-1e20)

check_and_fix_weights(quantized_model)

# Push the quantized model to the Hub
quantized_model.push_to_hub("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")
tokenizer.push_to_hub("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")

print("Quantized model pushed to the Hub successfully.")

# Save the quantized model locally Just in case
quantized_model.to("cpu")  # Move to CPU before saving if needed
quantized_model.save_pretrained("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")
tokenizer.save_pretrained("PeterAM4/EPFL-TA-Meister-GPTQ-4bit")

# Reload the quantized model to see if we can load it normally and use it with no issues
model = AutoModelForCausalLM.from_pretrained(
    "PeterAM4/EPFL-TA-Meister-GPTQ-4bit",
    device_map="auto"
)

# Example inference with the quantized model to check if it works
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}")
