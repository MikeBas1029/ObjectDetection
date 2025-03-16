import random
import os

def append_random_probabilities(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            random_prob = round(random.uniform(0, 1), 2)  # Generate probability with 2 decimal places
            outfile.write(f"{line} {random_prob}\n")

input_folder = "dataset/train/converted_images"
output_folder = "dataset/train/test_labels"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        append_random_probabilities(input_path, output_path)
        print(f"Processed: {filename}")
