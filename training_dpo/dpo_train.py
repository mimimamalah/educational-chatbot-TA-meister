from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback, DefaultFlowCallback
from datasets import load_dataset, concatenate_datasets
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
import os
import torch
import wandb
from dpo_callbacks import MultiMetricCallback 

os.environ["WANDB_PROJECT"] = "mnlp"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

# Initialize WandB
wandb.init(project="mnlp")

# Transform the new datasets to match the format of the DPO dataset
def transform_math_example(example):
    return {
        "prompt": example['instruction'],
        "chosen": example['chosen_response'],
        "rejected": example['rejected_response'],
    }
def transform_basic_example(example):
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": example['regected'],
    }

# Filter dataset to include only 'prompt', 'chosen' and 'rejected' fields
def filter_dataset(dataset):
    return dataset.map(lambda example: {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }).remove_columns([col for col in dataset.column_names if col not in ['prompt', 'chosen', 'rejected']])


# Set up access token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "./../checkpoints/checkpoint-5000"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# Load existing train and test datasets
train_dataset = load_dataset("json", data_files="./../datasets/training_m1/dpo_train_m1.jsonl")['train']
test_dataset = load_dataset("json", data_files="./../datasets/training_m1/dpo_test_m1.jsonl")['train']

# Load the additional stackexchange dataset 43k
stackexchange_dataset = load_dataset("json", data_files="./../datasets/dpo_stackexchange_43458.jsonl", split="train")

# Load the math DPO dataset and sample 2k examples
math_dataset = load_dataset("argilla/distilabel-math-preference-dpo", split="train").shuffle(seed=42).select(range(2000))

# Load the math DPO dataset and sample 9k examples
python_dataset = load_dataset("jondurbin/py-dpo-v0.1", split="train").shuffle(seed=42).select(range(9000))

# Load the STEM DPO dataset and sample 20k examples
stem_dataset = load_dataset("thewordsmiths/stem_dpo", split="train").shuffle(seed=42).select(range(20000))

# Transform and filter datasets
transformed_math_dataset = math_dataset.map(transform_math_example)
transformed_stem_dataset = stem_dataset.map(transform_basic_example)
transformed_python_dataset = python_dataset.map(transform_basic_example)

# Filter datasets to include only 'prompt', 'chosen', 'rejected' fields
train_dataset = filter_dataset(train_dataset)
stackexchange_dataset = filter_dataset(stackexchange_dataset)
transformed_math_dataset = filter_dataset(transformed_math_dataset)
transformed_stem_dataset = filter_dataset(transformed_stem_dataset)
transformed_python_dataset = filter_dataset(transformed_spython_dataset)

# Combine the datasets
combined_train_dataset = concatenate_datasets([
    train_dataset,
    stackexchange_dataset,
    transformed_math_dataset,
    transformed_stem_dataset,
    transformed_python_dataset
])

# Shuffle the combined dataset
combined_train_dataset = combined_train_dataset.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_id, attn_implementation="flash_attention_2", use_cache=False)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

# Set the LoRA configuration to target all linear layers
peft_config = LoraConfig(
    r=64,  # Set to 256 for higher rank
    lora_alpha=128,  # Corresponds to alpha for rsLoRA with rank 256 with formula
    lora_dropout=0.1,  # Typical dropout rate
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],  # Target all linear layers
    use_rslora=True  # Enable rsLoRA
)

training_args = DPOConfig(
    output_dir="./checkpoints/dpo",
    evaluation_strategy="steps",
    save_strategy="steps",
    # save_total_limit=5,                   # Limit the number of checkpoints to save Uncomment if needed
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_dir=f"./logs",  # Wandb logging directory
    logging_steps=10,  # Adjust the frequency of logging
    learning_rate=2e-6, # At least 10 x times smaller than the SFT
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=2,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper /10 because prevent unstable learning
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    report_to="wandb",                      # Enable WandB reporting
    load_best_model_at_end=True,            # Load the best model at the end of training
    metric_for_best_model="loss",           # 'loss' for simplicity
    greater_is_better=False,                # False for loss, True for accuracy, etc.
    torch_compile=True,                     # Enable TorchScript compilation
    tf32=True,                              # Use TF32 precision
    bf16=True,                              # Use BF16 precision
)

dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

callbacks = [
    EarlyStoppingCallback(early_stopping_patience=3),  # For early stopping
    MultiMetricCallback(),  # Custom callback for multiple metrics
    DefaultFlowCallback(),
]

trainer = DPOTrainer(
    model=model,
    ref_model=None, # set to none since we use peft
    args=training_args,
    train_dataset=combined_train_dataset,
    eval_dataset=test_dataset,  # Only use sft_test_m1.json for evaluation
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_length=1600,  # Chosen otherwise need too much RAM
    max_prompt_length=1600,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
    callbacks=callbacks, 
)

# Train the model
trainer.train()
