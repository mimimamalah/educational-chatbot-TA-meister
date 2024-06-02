from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

access_token = "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"
os.environ["HF_TOKEN"] = access_token

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    "./best_models/checkpoint_sft",
    low_cpu_mem_usage=True,
)  

# Merge LoRA and base model and save with a new name
merged_model = model.merge_and_unload()
new_model_name = "LLama-3-8B-SFT-4-pack"
merged_model.save_pretrained(new_model_name, safe_serialization=True, max_shard_size="5GB")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./best_models/checkpoint_sft")

# Save the tokenizer to the same directory as the merged model
tokenizer.save_pretrained(new_model_name)

print(f"Model and tokenizer saved to {new_model_name}")