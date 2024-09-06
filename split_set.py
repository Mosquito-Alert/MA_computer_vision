# Required libraries
import os
import shutil
from sklearn.model_selection import train_test_split

# Splits images from a folder into train, validation, and test folders.
# Parameters: image_folder, the path to the folder containing the images;
#             train_size, the proportion of the dataset to include in the train split;
#             val_size, the proportion of the dataset to include in the validation split.

def split_images(image_folder, train_size=0.7, val_size=0.15):

    # Check if image folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The folder '{image_folder}' does not exist.")
    
    # Create directories for train, validation and test splits
    train_dir = os.path.join(image_folder, 'train')
    val_dir = os.path.join(image_folder, 'val')
    test_dir = os.path.join(image_folder, 'test')
    
    # Create the directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get list of all images
    images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    # First split into train and temp (validation + test)
    train_images, temp_images = train_test_split(images, train_size=train_size, random_state=42)
    
    # Calculate the relative size for validation out of the remaining dataset
    val_relative_size = val_size / (1 - train_size)
    
    # Split temp into validation and test
    val_images, test_images = train_test_split(temp_images, train_size=val_relative_size, random_state=42)
    
    # Helper function to copy images to the respective directories
    def copy_images(image_list, destination_dir):
        for image in image_list:
            src = os.path.join(image_folder, image)
            dst = os.path.join(destination_dir, image)
            shutil.copy(src, dst)
    
    # Copy images to respective directories
    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)
    copy_images(test_images, test_dir)
    
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

# The script should be saved in the same directory as 'images' folder.
split_images("images")