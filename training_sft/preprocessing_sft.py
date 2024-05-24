import json

input_file = 'M1_processed_data.json'  # replace with your input file path
output_file = 'M1_sft.json'  # replace with your output file path

# Read the input data
with open(input_file, 'r') as f:
    data = json.load(f)

# Transform the data
formatted_data = []
for item in data:
    formatted_data.append({
        "prompt": item["input"],
        "completion": item["label"]
    })

# Write the transformed data to the output file
with open(output_file, 'w') as f:
    for item in formatted_data:
        json.dump(item, f)
        f.write('\n')
