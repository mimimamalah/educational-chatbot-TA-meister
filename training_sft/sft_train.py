from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from peft import LoraConfig
import os
import torch

os.environ["WANDB_PROJECT"] = "mnlp"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

# # Function to get all the linear layers
# def get_linear_layers(model):
#     linear_layers = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             linear_layers.append(name)
#     return linear_layers

# Transform the new datasets to match the format of the SFT dataset
def transform_example(example):
    return {
        "prompt": f"Question: {example['problem']}",
        "completion": example['solution']
    }

def transform_physics_example(example):
    return {
        "prompt": example["message_1"],
        "completion": example["message_2"]
    }

def transform_code_example(example):
    return {
        "prompt": example["query"],
        "completion": example["answer"]
    }

# Filter dataset to include only 'prompt' and 'completion' fields
def filter_dataset(dataset):
    return dataset.map(lambda example: {
        "prompt": example["prompt"],
        "completion": example["completion"]
    }).remove_columns([col for col in dataset.column_names if col not in ['prompt', 'completion']])

# Set up access token and model ID
access_token = "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"
os.environ["HF_TOKEN"] = access_token
model_id = "meta-llama/Meta-Llama-3-8B"

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# Load existing train and test datasets
train_dataset = load_dataset("json", data_files="./../datasets/training_m1/sft_train_m1.json")['train']
test_dataset = load_dataset("json", data_files="./../datasets/training_m1/sft_test_m1.json")['train']

# Load the new competition_math dataset
new_train_dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
new_test_dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

# Load the additional stackexchange dataset
stackexchange_dataset = load_dataset("json", data_files="./../datasets/sft_stackexchange_43043.json", split="train")

# Load the physics dataset and sample 10k examples
physics_dataset = load_dataset("lgaalves/camel-ai-physics", split="train").shuffle(seed=42).select(range(10000))

# Load the code feedback dataset and sample 10k examples
code_feedback_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train").shuffle(seed=42).select(range(10000))

# Transform and filter datasets
transformed_new_train_dataset = new_train_dataset.map(transform_example)
transformed_new_test_dataset = new_test_dataset.map(transform_example)
transformed_physics_dataset = physics_dataset.map(transform_physics_example)
transformed_code_feedback_dataset = code_feedback_dataset.map(transform_code_example)

# Filter datasets to include only 'prompt' and 'completion' fields
train_dataset = filter_dataset(train_dataset)
stackexchange_dataset = filter_dataset(stackexchange_dataset)
transformed_new_train_dataset = filter_dataset(transformed_new_train_dataset)
transformed_new_test_dataset = filter_dataset(transformed_new_test_dataset)
transformed_physics_dataset = filter_dataset(transformed_physics_dataset)
transformed_code_feedback_dataset = filter_dataset(transformed_code_feedback_dataset)

# Combine the datasets
combined_train_dataset = concatenate_datasets([
    train_dataset,
    transformed_new_train_dataset,
    transformed_new_test_dataset,
    stackexchange_dataset,
    transformed_physics_dataset,
    transformed_code_feedback_dataset
])

# Shuffle the combined dataset
combined_train_dataset = combined_train_dataset.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", use_cache=False)

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

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="steps",
    save_strategy="steps",
    # save_total_limit=5,                   # Limit the number of checkpoints to save Uncomment if needed
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_dir=f"./logs",  # Wandb logging directory
    logging_steps=10,  # Adjust the frequency of logging
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=2,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.03,                     # max gradient norm based on QLoRA paper /10 because prevent unstable learning
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    report_to="wandb",                      # Enable WandB reporting
    load_best_model_at_end=True,            # Load the best model at the end of training
    metric_for_best_model="loss",           # 'loss' for simplicity
    greater_is_better=False,                # False for loss, True for accuracy, etc.
    torch_compile=True,                     # Enable TorchScript compilation
    tf32=True,                              # Use TF32 precision
    bf16=True,                              # Use BF16 precision
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_dataset,
    eval_dataset=test_dataset,  # Only use sft_test_m1.json for evaluation
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=1600,  # Chosen otherwise need too much RAM
)

# Train the model
trainer.train()
