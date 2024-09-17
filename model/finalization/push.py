import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Peter's API token, type: write
os.environ["HF_TOKEN"] = "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"

# Model path with repo name
model_id = "./LLama-3-8B-SFT-4-pack"
model_repo_name = "PeterAM4/EPFL-TA-Meister-SFT"  # Replace with your Hugging Face username and desired model name

# As always
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Push to Hub
model.push_to_hub(model_repo_name, use_auth_token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(model_repo_name, use_auth_token=os.getenv("HF_TOKEN"))
