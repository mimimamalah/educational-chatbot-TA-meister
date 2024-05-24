"""
This python script merges multiple DPO preference pairs files into one
"""
import argparse
import jsonlines
import random
import glob
import os

def read_jsonl_file(file_path: str) -> list[str]:
    """Read lines from a JSONL file."""
    lines = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            lines.append(line)
    return lines

def write_jsonl_file(file_path: str, lines: list[str]):
    """Write lines to a JSONL file."""
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(lines)

def select_random_lines(lines: list[str], num_lines: int) -> list[str]:
    """Select a specified number of random elements from a list."""
    if len(lines) <= num_lines:
        return lines
    
    return random.sample(lines, num_lines)

def process_jsonl_files(input_dir: str, output_file: str, num_lines: int, num_lines_phy_stack):
    """Process multiple JSONL files and write random lines to the final JSONL file."""
    all_selected_lines = []
    
    # Get a list of all files matching the input pattern
    input_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    print(f"Input files: {input_files}")
    
    for file in input_files:
        print(f"\nProcessing {file}")
        print("=" * 50)
        lines = read_jsonl_file(file)
        print(f"Read {len(lines)} lines")

        if "stackoverflow" in str(file) or "physics" in str(file):
            selected_lines = select_random_lines(lines, num_lines_phy_stack)
        else:
            selected_lines = select_random_lines(lines, num_lines)

        print(f"Selected {len(selected_lines)} random lines")

        all_selected_lines.extend(selected_lines)
    
    write_jsonl_file(output_file, all_selected_lines)
    print(f"\nSuccessfully written {len(all_selected_lines)} lines to {output_file}")


if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser(description='Data Merging Script')
    parser.add_argument('num_lines', type=int, help='The number of lines to randomly take from each jsonl file')
    parser.add_argument('num_lines_phy_stack', type=int, help='The number of lines to randomly take from the stack'
                        ' overflow and physics dataset. We treat these dataset differently as their data is considered'
                        ' to be of lower quality.')
    parser.add_argument('input_directory', type=str, help='The directory containing all the jsonl files to merge')
    parser.add_argument('output_file', type=str, help='Path to the merged output file')
        
    args = parser.parse_args()

    process_jsonl_files(args.input_directory, args.output_file, args.num_lines, args.num_lines_phy_stack)
