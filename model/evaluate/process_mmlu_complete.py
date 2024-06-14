from datasets import load_dataset
import json
from process_mmlu import load_multiple_datasets, transform_entry

task_list = [
    "high_school_physics",
    "college_computer_science",
    "abstract_algebra",
    "machine_learning",
    "college_mathematics",
    "formal_logic",
    "high_school_statistics",
    "high_school_mathematics",
    "high_school_computer_science",
    "college_physics",
    "computer_security",
]

# Load multiple datasets
datasets_dict = load_multiple_datasets(task_list)

all_transformed_entries = []
for task_name, dataset in datasets_dict.items():
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            transformed_entries = [transform_entry(entry, task_name) for entry in dataset[split]]
            all_transformed_entries.extend(transformed_entries)


# Write to a single JSONL file
with open("model/datasets/mcqa_final.jsonl", 'w') as outfile:
    for entry in all_transformed_entries:
        json.dump(entry, outfile)
        outfile.write('\n')