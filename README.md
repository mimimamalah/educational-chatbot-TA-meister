[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/xFAY6za9)
# Llama3-8B x EPFL

Welcome to our MNLP repository! Here's a brief overview of the project structure and key components:


## Contents:
**model directory:** Contains evaluation datasets and benchmarks

**data directory:** Contains the training data and the data processing scripts.

**finalization directory:** Contains scripts that merges LoRA weights and pushes the final model to hugging face.

**sft_train.py:** Script containing the SFT training pipeline.

**dpo_train.py:** Script containing the DPO training pipeline.

## Model Availability
Our final aligned model (SFT + DPO) is available on hugging face under the identifier: "PeterAM4/EPFL-TA-Meister"

The finetuned model that we used prior to DPO (SFT only) is available under the identifier: PeterAM4/EPFL-TA-Meister-SFT
