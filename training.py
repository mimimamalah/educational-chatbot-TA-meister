import torch
import os
if torch.cuda.is_available():
    print("GPU is available! Good to go.")
else:
    print(
        "If you are using Colab, please set your runtime type to a GPU via {Edit -> Notebook Settings}."
    )

seed = 42
torch.manual_seed(seed)
torch.set_printoptions(precision=16) # set print number precision at 16

from models import model_base
import transformers
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B"
access_token = "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"
os.environ["HF_TOKEN"] = access_token

from transformers.utils.quantization_config import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
 load_in_4bit=False,
 bnb_4bit_quant_type="nf4",
 bnb_4bit_compute_dtype=torch.bfloat16,
)

loaded_model = AutoModelForCausalLM.from_pretrained(model_id)

from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(r=128, lora_alpha=16, target_modules='all-linear', lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
base_model = get_peft_model(loaded_model, peft_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset
train_dataset = load_dataset('json', data_files='datasets/dpo_preference_example.jsonl',split='train').train_test_split(test_size=0.35)
train_dataset = train_dataset['test']
eval_dataset = load_dataset('json', data_files='datasets/dpo_preference_example.jsonl',split='train').train_test_split(test_size=0.002)
eval_dataset = eval_dataset['test']

from trl import DPOConfig, DPOTrainer

args = DPOConfig(
    output_dir="checkpoints",             
    num_train_epochs=1,                    # number of training epochs
    per_device_train_batch_size=1,         # batch size per device during training
    per_device_eval_batch_size=1,           # batch size for evaluation
    gradient_accumulation_steps=5,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=400,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

max_seq_length = 1024
prompt_length = 1024

from evaluate_bert import BERTScoreEvaluator

trainer = DPOTrainer(
    base_model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
    callbacks=[BERTScoreEvaluator(tokenizer)] # need to write the evaluator
)

trainer.train()
