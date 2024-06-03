"""
Script that transforms dpo preference data to sft training data by selecting the prompt along the "chosen" response.
"""
import argparse
import jsonlines
import json

def transform_jsonl_to_json(input_file, output_file):
    transformed_data = []
    
    # Read the input JSONL file
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            transformed_obj = {
                "prompt": obj["prompt"],
                "completion": obj["chosen"]
            }
            transformed_data.append(transformed_obj)
    
    # Write the transformed data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)
    
    print(f"Successfully transformed {len(transformed_data)} records from {input_file} to {output_file}")

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser(description='Data Merging Script')
    parser.add_argument('input', type=str, help='The jsonl DPO input file')
    parser.add_argument('output', type=str, help='Path to the json SFT output file')
        
    args = parser.parse_args()

    transform_jsonl_to_json(args.input, args.output)
