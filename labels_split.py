import os
import shutil

# Define paths for image splits and labels
image_splits = {
    'train': './images/train',
    'val': './images/val',
    'test': './images/test'
}
labels_dir = './labels'

# Create corresponding directories for label splits
os.makedirs(f'{labels_dir}/train', exist_ok=True)
os.makedirs(f'{labels_dir}/val', exist_ok=True)
os.makedirs(f'{labels_dir}/test', exist_ok=True)

# Function to split labels according to images
def split_labels():
    for split, img_dir in image_splits.items():
        # List all images in the current split directory
        image_files = os.listdir(img_dir)
        
        # For each image, find its corresponding label and move it to the appropriate label folder
        for img_file in image_files:
            img_name, img_ext = os.path.splitext(img_file)
            label_file = f'{img_name}.txt'  # Label file with the same name as the image
            
            label_src_path = os.path.join(labels_dir, label_file)  # Original label path
            label_dest_path = os.path.join(labels_dir, split, label_file)  # Destination path

            # Check if the label file exists, and move it to the respective directory
            if os.path.exists(label_src_path):
                shutil.move(label_src_path, label_dest_path)
                print(f"Moved: {label_src_path} -> {label_dest_path}")
            else:
                print(f"Label not found for image: {img_file}")

if __name__ == '__main__':
    split_labels()
    print("Label splitting completed.")
