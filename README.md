[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/NXyI9KHk)

### Note: Following our discussion on Ed #1099, we modified only the README after the deadline.

# CS-552 - Final submission

## Our models :

"PeterAM4/EPFL-TA-Meister" : our final model with SFT and DPO training  
"PeterAM4/EPFL-TA-Meister-SFT" : our final model with SFT only  
"PeterAM4/EPFL-TA-Meister-GPTQ-4bit" : GPTQ quantized model  
"PeterAM4/EPFL-TA-Meister-4bit" : bites and bytes quantized model 4 bit  
"PeterAM4/EPFL-TA-Meister-8bit" : bites and bytes quantized model 8 bit  
"PeterAM4/EPFL-TA-Meister-AWQ-4bit" : AWQ quantized model   


## Codebase File Structure

```txt
.
── _templates
│   └── CS-552 Final Report Template.zip
├── _tests
│   ├── model_files_validator.py
│   ├── model_rag_validator.py
│   └── pdf_report_validator.py
├── pdfs
│   └── 4-pack-M3.pdf
├── model
│   ├── data [data for evaluation]
│       ├── sft_stackexchange_43043.json
│   │   └── sft_train_m1.json
│   ├── datasets
│   │   └── All datasets used for evaluation in jsonl files
│   ├── documents [FOR RAG ONLY]
│   │   └── STEM Books as PDF files
│   ├── evaluate [scripts for dataset evaluation]
│   │       ├── mmlu.py
│   │       ├── process_mmlu_complete.py
│   │       ├── process_mmlu_subset.py
│   │       └── process_mmlu.py
│   ├── models
│   │       ├── model_base.py
│   │       ├── model_dpo_utils.py
│   │       └── model_dpo.py
│   ├── quantization [scripts we used to quantize]
│   │       ├── data_processing.py
│   │       ├── quantization_AWQ.py
│   │       ├── quantization_bitsandbytes_4bit.py
│   │       ├── quantization_bitsandbytes_8bit.py
│   │       └── data_processing.py

│   ├── utils.py
│   ├── evaluator.py
│   ├── main_config.yaml
│   ├── populate_database.py
│   ├── README.md
│   ├── requirements.txt
└── README.md
```

## Contributions
Peter Abdel Massih and Malak Lahlou Nabil: DPO training, Quantization, Running evaluation.   
Frederic Khayat and Sara Anejjar: Datasets generation, Evaluation metrics, Evaluator adaption, RAG.
