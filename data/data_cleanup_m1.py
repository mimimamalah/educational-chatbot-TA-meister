"""
This python script transforms the M1 data (json) into DPO preference pair format (jsonl)
"""
import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
import utils


def generate_preference_pairs(input_data: list[dict]) -> list[dict]:
    """
    Generate DPO preference pairs from the M1 data.
    """
    len_input = sum([len(sample['preference']) for sample in input_data])
    len_output = 0
    len_ignored = 0

    
    preference_pairs = []
    for sample in input_data:
        prompt = sample['question_complete']

        for preference in sample['preference']:
            # Skip preference pairs that do not have an answer
            if preference['A'] == "..." or preference['B'] == "...":
                len_ignored += 1
                continue

            assert(preference['overall'] in ['A', 'B'])
            chosen = preference['overall']
            rejected = 'A' if chosen == 'B' else 'B'
            preference_pairs.append({
                'prompt': prompt,
                'chosen': preference[chosen],
                'rejected': preference[rejected]
            })
            len_output += 1

    assert(len_output + len_ignored == len_input)
    return preference_pairs


if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser(description='M1 Data Processing Script')
    parser.add_argument('input', type=str, help='Path to input file')
    parser.add_argument('output', type=str, help='Path to output file')
        
    args = parser.parse_args()

    # Read unprocessed data, filter out invalid questions, and write to output file
    input_data = utils.read_json(args.input)
    print("Read", len(input_data), "questions\n")

    output_data = generate_preference_pairs(input_data)
    print("Generated", len(output_data), "preference pairs\n")

    utils.write_jsonl(output_data, args.output)
    print("Data written to", args.output, "successfully\n")
