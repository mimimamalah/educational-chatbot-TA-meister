from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# Set up your Hugging Face API token
os.environ["HF_TOKEN"] = "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"

# Define the configuration for 8-bit quantization with an outlier threshold
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0  # Set outlier threshold to 6.0 changed to fp16
)

# Load the model with quantization high precision layers are still in fp32
model = AutoModelForCausalLM.from_pretrained(
    "PeterAM4/EPFL-TA-Meister", 
    quantization_config=quantization_config
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("PeterAM4/EPFL-TA-Meister")

# Push the quantized model to the Hugging Face Hub
model.push_to_hub("PeterAM4/EPFL-TA-Meister-8bit")
tokenizer.push_to_hub("PeterAM4/EPFL-TA-Meister-8bit")
