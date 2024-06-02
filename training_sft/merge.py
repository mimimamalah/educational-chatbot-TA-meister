from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    "./best_models/checkpoint_sft",
    low_cpu_mem_usage=True,
)  

# Merge LoRA and base model and save with a new name
merged_model = model.merge_and_unload()
new_model_name = "LLama-3-8B-SFT-4-pack"
merged_model.save_pretrained(new_model_name, safe_serialization=True, max_shard_size="5GB")
