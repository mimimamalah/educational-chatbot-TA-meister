import jsonlines

def calculate_max_lengths(file_path):
    max_prompt_length = 0
    max_completion_length = 0
    max_sequence_length = 0
    exceeding_max_length_count = 0
    max_seq_length = 2500  # or any length you want to check for

    with jsonlines.open(file_path) as reader:
        for entry in reader:
            prompt_length = len(entry['prompt'])
            completion_length = len(entry['chosen'])
            sequence_length = prompt_length + completion_length
            
            if prompt_length > max_prompt_length:
                max_prompt_length = prompt_length
                
            if completion_length > max_completion_length:
                max_completion_length = completion_length
                
            if sequence_length > max_sequence_length:
                max_sequence_length = sequence_length
            
            if sequence_length > max_seq_length:
                exceeding_max_length_count += 1
    
    return max_prompt_length, max_completion_length, max_sequence_length, exceeding_max_length_count

def main():
    # Path to the JSONL file
    file_path = './dpo_m1.jsonl'
    
    max_prompt_length, max_completion_length, max_sequence_length, exceeding_max_length_count = calculate_max_lengths(file_path)
    
    print(f"Maximum prompt length: {max_prompt_length} characters")
    print(f"Maximum completion length: {max_completion_length} characters")
    print(f"Maximum sequence length: {max_sequence_length} characters")
    print(f"Number of sequences exceeding 512 characters: {exceeding_max_length_count}")

if __name__ == '__main__':
    main()
