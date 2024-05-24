from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import os
import torch

access_token = "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"
os.environ["HF_TOKEN"] = access_token

model_id = "meta-llama/Meta-Llama-3-8B"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

dataset = load_dataset("json", data_files="M1_sft.json", split="train")
dataset = dataset.train_test_split(test_size=0.1)  # 10% of the data will be used for testing
train_dataset = dataset['train']
test_dataset = dataset['test']

tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj",
            "k_proj",
            "v_proj",
            "o_proj"]
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_total_limit=5,
    save_steps=500,
    gradient_accumulation_steps=4,
    optim = "adamw_8bit",
    gradient_checkpointing=True,
    bf16=True,
    logging_dir=f"./logs",  # TensorBoard logging directory
    logging_steps=10,  # Adjust the frequency of logging
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="tensorboard",  # Enable TensorBoard reporting
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config
)

# Train the model
trainer.train()
