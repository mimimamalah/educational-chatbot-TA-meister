import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os

os.environ["HF_TOKEN"] = "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"

output_file = 'M1_processed_data.json'

# Load the processed dataset
dataset = load_dataset('json', data_files=output_file)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set max_length
max_length = 8192  # Adjust based on your model's capability

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(examples['label'], padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": inputs["input_ids"], 
        "attention_mask": inputs["attention_mask"], 
        "labels": labels["input_ids"]
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset
split_datasets = tokenized_dataset['train'].train_test_split(test_size=0.1)
train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

# Load the model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,  # Adjust this value based on your specific needs and GPU capacity
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=5,
    per_device_train_batch_size=2,  # Adjust batch size based on available GPU memory
    per_device_eval_batch_size=2,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False, ## because using eval loss
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine_tuned_model_lora')