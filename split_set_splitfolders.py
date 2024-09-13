# Required libraries
import os
import splitfolders  # To install: pip install split-folders

# Splits images from a folder into train, validation, and test folders.
# Parameters: image_folder, the path to the folder containing the images;
#             train_size, the proportion of the dataset to include in the train split;
#             val_size, the proportion of the dataset to include in the validation split.

def split_images(image_folder, train_size=0.7, val_size=0.15):

    # Check if image folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The folder '{image_folder}' does not exist.")
    
    # Define output directory
    output_folder = os.path.join(image_folder, 'output')
    
    # Ensure that train_size and val_size are correctly specified
    if train_size + val_size > 1.0:
        raise ValueError("The sum of train_size and val_size should not exceed 1.0.")

    # Remaining size for test set
    test_size = 1.0 - train_size - val_size

    # Split the dataset using split-folders library
    splitfolders.ratio(image_folder, output=output_folder, seed=42, 
                       ratio=(train_size, val_size, test_size), 
                       group_prefix=None, move=False)  # Set move=True if you want to move files
    
    print(f"Images have been successfully split into train, validation, and test sets.")
    
# The script should be saved in the same directory as the 'images' folder.
split_images("images")
