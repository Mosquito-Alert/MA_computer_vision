# This script crops all images in 'images' folder according to the bounding boxes specified in
# 'labels', which should be written in YOLOv8 format. Cropped images are saved in 'cropped_images' folder.
# The script should be saved in the same directory as 'images' and 'labels' folders.
# Finally, information on running time and performance is included in 'cropping_info.txt'.

# Required libraries
import os
import cv2
import time

# Create a new folder for the cropped images
if not os.path.exists('cropped_images'):
    os.makedirs('cropped_images')

# Iterate over all images
images = os.listdir('images')

start_time = time.time()
errors = 0

for image in images:
    img_path = 'images/' + image # Relative path to the image
    img_label_path = 'labels/' + image.rsplit('.', 1)[0] + '.txt' # Relative path to the label

    try:
        # Read info from the label
        with open(img_label_path) as f:
            info = f.read().strip() # Ignore trailing spaces
            coord = info.split() # Coordinates are delimited by blank spaces

        # Load image    
        img = cv2.imread(img_path)

        # Check that image is correctly loaded
        if img is None:
            print(f"Error while loading {image}.")
            continue

        # Denormalize coordinates
        height, width = img.shape[:2]

        center_x = float(coord[1]) * width
        center_y = float(coord[2]) * height

        rect_width = float(coord[3]) * width
        rect_height = float(coord[4]) * height

        # Compute necessary vertices of the crop region
        top_left = (int(center_x - rect_width / 2), int(center_y - rect_height / 2))
        bottom_right = (int(center_x + rect_width / 2), int(center_y + rect_height / 2))

        # Crop the image
        cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Save cropped image
        cv2.imwrite('cropped_images/' + image, cropped_img)

        print(image + ' cropped successfully.')

    except Exception as e:
        errors += 1
        print(f'Error while cropping {image}: {e}.')

end_time = time.time()

t = end_time - start_time
l = len(images)

with open('cropping_info.txt', 'w') as f:
    f.write(f"Successfully cropped images: {l - errors}/{l}.\n")
    f.write(f"Total running time: {round(t / 60,2)} min. Average time per cropped image: {round(t / (l - errors), 4)} s.")
