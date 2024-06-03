from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

access_token = "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"
os.environ["HF_TOKEN"] = access_token

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    "./model/checkpoints/checkpoint-12000",
    low_cpu_mem_usage=True,
)  

# Load the tokenizer, since it should also be in the same directory
tokenizer = AutoTokenizer.from_pretrained("./model/checkpoints/checkpoint-12000")

# Now we merge the model under a new name while keeping shards under 5GB save it to the current directory with the tokenizer
merged_model = model.merge_and_unload()
new_model_name = "LLama-3-8B-SFT-4-pack"
merged_model.save_pretrained(new_model_name, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(new_model_name)

print(f"Model and tokenizer saved to {new_model_name}")