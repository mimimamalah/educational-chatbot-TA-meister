import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up your Hugging Face API token
os.environ["HF_TOKEN"] = "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"

# Define your model path and Hugging Face repository name
model_id = "./LLama-3-8B-SFT-4-pack"
model_repo_name = "PeterAM4/EPFL-TA-Meister-SFT"  # Replace with your Hugging Face username and desired model name

# Load your trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(model_repo_name, use_auth_token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(model_repo_name, use_auth_token=os.getenv("HF_TOKEN"))
