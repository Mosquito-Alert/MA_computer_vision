import os
import re

def replace_first_digit(input_folder, output_folder):

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only .txt files
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            
            # Open the input file
            with open(input_file_path, 'r') as file:
                content = file.read()
                
            # Replace the first digit with '0'
            modified_content = re.sub(r'\d', '0', content, count=1)
            
            # Write modified content to the output file
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w') as file:
                file.write(modified_content)

input_folder = r"C:\Users\damuk\Desktop\JAE Intro\mosquito_comp_vision_v3\train\labels"
output_folder = r"C:\Users\damuk\Desktop\JAE Intro\mosquito_comp_vision_v3\train\labels_single"

replace_first_digit(input_folder, output_folder)
