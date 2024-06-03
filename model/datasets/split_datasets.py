"""
This script splits a DPO (Dialogue Policy Optimization) dataset and a corresponding SFT (Supervised Fine-Tuning) dataset 
into training and test sets. The test sets for both datasets are ensured to have matching prompts.

The script performs the following steps:
1. Reads the DPO dataset from a JSONL file and selects 3% of the data randomly as the test set.
2. Splits the DPO dataset into training and test sets based on the selected indices.
3. Reads the SFT dataset from a JSON file and splits it into training and test sets using the same indices as the DPO test set.
4. Writes the training and test sets for both datasets to separate JSON and JSONL files.

Usage:
    python split_datasets.py dpo_input.jsonl sft_input.json dpo_train.jsonl dpo_test.jsonl sft_train.json sft_test.json

Arguments:
    dpo_input: The input JSONL file for the DPO dataset.
    sft_input: The input JSON file for the SFT dataset.
    dpo_output_train: The output JSONL file for the DPO training set.
    dpo_output_test: The output JSONL file for the DPO test set.
    sft_output_train: The output JSON file for the SFT training set.
    sft_output_test: The output JSON file for the SFT test set.
"""

import argparse
import json
import jsonlines
import random

def split_dataset(input_file, output_train_file, output_test_file, test_size=0.03):
    with jsonlines.open(input_file) as reader:
        data = list(reader)
    
    test_count = int(len(data) * test_size)
    test_indices = set(random.sample(range(len(data)), test_count))
    
    train_data = [data[i] for i in range(len(data)) if i not in test_indices]
    test_data = [data[i] for i in range(len(data)) if i in test_indices]
    
    with jsonlines.open(output_train_file, 'w') as writer:
        writer.write_all(train_data)
    
    with jsonlines.open(output_test_file, 'w') as writer:
        writer.write_all(test_data)
    
    return test_indices

def filter_sft_data(sft_file, output_train_file, output_test_file, test_indices):
    with open(sft_file) as f:
        data = json.load(f)
    
    train_data = [data[i] for i in range(len(data)) if i not in test_indices]
    test_data = [data[i] for i in range(len(data)) if i in test_indices]
    
    with open(output_train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(output_test_file, 'w') as f:
        json.dump(test_data, f, indent=4)

def main(dpo_input, sft_input, dpo_output_train, dpo_output_test, sft_output_train, sft_output_test):
    test_indices = split_dataset(dpo_input, dpo_output_train, dpo_output_test)
    filter_sft_data(sft_input, sft_output_train, sft_output_test, test_indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split DPO and SFT datasets into train and test sets')
    parser.add_argument('dpo_input', type=str, help='The input JSONL file for the DPO dataset')
    parser.add_argument('sft_input', type=str, help='The input JSON file for the SFT dataset')
    parser.add_argument('dpo_output_train', type=str, help='The output JSONL file for the DPO training set')
    parser.add_argument('dpo_output_test', type=str, help='The output JSONL file for the DPO test set')
    parser.add_argument('sft_output_train', type=str, help='The output JSON file for the SFT training set')
    parser.add_argument('sft_output_test', type=str, help='The output JSON file for the SFT test set')
    
    args = parser.parse_args()
    
    main(args.dpo_input, args.sft_input, args.dpo_output_train, args.dpo_output_test, args.sft_output_train, args.sft_output_test)
