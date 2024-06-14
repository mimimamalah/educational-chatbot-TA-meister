from datasets import load_dataset
import json

# Function to load multiple datasets
def load_multiple_datasets(task_list):
    datasets_dict = {}
    for task_name in task_list:
        try:
            dataset = load_dataset('evaluate/mmlu.py', name=task_name, trust_remote_code=True)
            datasets_dict[task_name] = dataset
        except Exception as e:
            print(f"Error loading dataset for task {task_name}: {e}")
    return datasets_dict

def transform_entry(entry, subject):
    question_text = f"Question: {entry['input']}\n\nOptions:\n"
    options_text = "".join(
        f"{chr(65 + idx)}. {entry[chr(65 + idx)]}\n" for idx in range(4)
    )
    full_question = question_text + options_text + "\nAnswer:"
    transformed_data = {
        'subject': subject,
        'question': full_question,
        'answer': entry['target']
    }
    return transformed_data

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