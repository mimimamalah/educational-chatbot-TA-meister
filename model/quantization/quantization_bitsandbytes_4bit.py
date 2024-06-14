from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os 

# Set up access token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "PeterAM4/EPFL-TA-Meister"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load the model in 4-bit
    bnb_4bit_use_double_quant=True,       # Use double quantization
    bnb_4bit_quant_type="nf4",            # Use the nf4 quantization type weights from normal distribution
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation of the quantized weights
)

# Load model and tokenizer
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Push the quantized model to the Hugging Face Hub
quantized_model.push_to_hub("PeterAM4/EPFL-TA-Meister-4bit")
tokenizer.push_to_hub("PeterAM4/EPFL-TA-Meister-4bit")


