from datasets import load_dataset

# Function to load multiple datasets
def load_multiple_datasets(task_list):
    datasets_dict = {}
    for task_name in task_list:
        try:
            dataset = load_dataset('model/evaluate/mmlu.py', name=task_name, trust_remote_code=True)
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
