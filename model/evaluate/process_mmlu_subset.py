from datasets import load_dataset
import json
import os
import random
from process_mmlu import load_multiple_datasets, transform_entry

task_list = [
    "college_computer_science",
    "abstract_algebra",
    "machine_learning",
    "college_mathematics",
    "formal_logic",
    "college_physics",
]

# Load multiple datasets
datasets_dict = load_multiple_datasets(task_list)

for task_name, dataset in datasets_dict.items():
    all_transformed_entries = []
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            transformed_entries = [transform_entry(entry, task_name) for entry in dataset[split]]
            all_transformed_entries.extend(transformed_entries)
    
    # Shuffle and select up to 60 entries
    random.shuffle(all_transformed_entries)
    selected_entries = all_transformed_entries[:60]
    
    # Write to a separate JSONL file for each task
    with open(f"model/datasets/mcqa_{task_name}.jsonl", 'w') as outfile:
        for entry in selected_entries:
            json.dump(entry, outfile)
            outfile.write('\n')
