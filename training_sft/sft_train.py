from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import os
import torch

access_token = "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"
os.environ["HF_TOKEN"] = access_token

# model_id = "meta-llama/Meta-Llama-3-8B"
model_id = "./best_models/checkpoint_sft_m1"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

dataset = load_dataset("json", data_files="sft_stackexchange_43043.json", split="train")
dataset = dataset.train_test_split(test_size=0.05)  # 5% of the data will be used for testing
train_dataset = dataset['train']
test_dataset = dataset['test']

tokenizer = AutoTokenizer.from_pretrained(model_id, attn_implementation="flash_attention_2")
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj",
            "k_proj",
            "v_proj",
            "o_proj"],
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=5,
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bf16=True,
    logging_dir=f"./logs",  # TensorBoard logging directory
    logging_steps=10,  # Adjust the frequency of logging
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    report_to="tensorboard",  # Enable TensorBoard reporting
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="loss",  # 'loss' for simplicity
    greater_is_better=False,  # False for loss, True for accuracy, etc.
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=1600, # Chosen otherwise need too much RAM
)

# Train the model
trainer.train()
