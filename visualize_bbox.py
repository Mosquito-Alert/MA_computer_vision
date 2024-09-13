# Required libraries
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to denormalize bounding box coordinates
def denormalize_bbox(bbox, img_width, img_height):
    center_x, center_y, width, height = bbox
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    bottom_right_x = int(center_x + width / 2)
    bottom_right_y = int(center_y + height / 2)
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

# Function to draw bounding box on image
def draw_bbox(image_path, bbox):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    top_left, bottom_right = denormalize_bbox(bbox, width, height)
    
    cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0), thickness=2)
    return img

# Function to read YOLOv8 labels from a text file
def read_label(label_path):
    with open(label_path, 'r') as f:
        line = f.readline().strip().split()  # Read only the first line (one bounding box)
    bbox = [float(val) for val in line[1:]]  # Skip the class label and get coordinates
    return bbox

# Function to visualize random bounding boxes from a batch of images
def visualize_random_bboxes(image_folder, label_folder, num_images=10):
    # Get list of image files and corresponding label files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f.replace('.jpeg', '.txt').replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # Randomly select a subset of images
    selected_images = random.sample(image_files, num_images)
    
    # Plot the selected images with bounding boxes
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns for 10 images
    axs = axs.flatten()
    
    for i, img_file in enumerate(selected_images):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, label_files[image_files.index(img_file)])
        
        # Read bounding box information
        bboxes = read_label(label_path)
        
        # Plot the image with bounding box
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw all bounding boxes on the image
        for bbox in bboxes:
            top_left, bottom_right = denormalize_bbox(bbox, img.shape[1], img.shape[0])
            cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0), thickness=2)
        
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.show()

# Visualize 10 random images with bounding boxes
visualize_random_bboxes('images', 'labels')
