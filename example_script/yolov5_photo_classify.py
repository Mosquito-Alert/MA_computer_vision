# Script for testing trained model
# of yolov5 against a sigle of photo
# or set of photos
# taken by mobile telephone
# without any class associated
# with the photo
# could be upladed to the web
# to the cloud 
# and return a reponse 
# with assigned class and probability score

# Install and Import Dependencies
import torch
import os
from pathlib import Path
import cv2
import csv
from timeit import default_timer as timer
from difflib import SequenceMatcher

from matplotlib import pyplot as plt
import numpy as np
import glob
from IPython.display import Image, display

try:
    from PIL import Image
except ImportError:
    import Image


# get key from dictionary by given value 
# for getting infomration about class number from class name
def get_key_by_value_dict(expert_class):
    found_key = None
    for key, value in class_labels.items():
        if expert_class == value:
            found_key = key
    return found_key


# classifying image by yolov5
# (mosquito class and bounding box )
# what if there is an image with more 
# mosquitos in the image
# returned results should look like:
# results.pandas().xyxy[0] 
# 	xmin	ymin	xmax	ymax	confidence	class	name
#0	187.451035	192.202072	681.199463	675.649109	0.705023	1	albopictus
def classify_image(image):
    image_information = {}
    result = yolov5_model(image)
    result_df = result.pandas().xyxy[0]
    if result_df.empty:
        print('No results from yolov5 model!')
    else:
        image_information = result_df.to_dict()
    # for detection in result:
    #     text =   detection[1][0]
    #     lines += text + "\n"

    # uncomment if you want to see or /and save the images of mosquito and bboxes and probalility
    # result.show()
    print(result_df)
    result.save()
    return image_information


# checks if mosquito class (manually classified)
# is present in image information
def check_mosquito_class(mosquito_class_predicted, mosquito_class_entomologists):
    correct_prediction = False
    if mosquito_class_entomologists == mosquito_class_predicted:
        correct_prediction = True
    return correct_prediction

# mosquito_class number is always written as a first part of the text
# and divided by space " " from the rest of the
# mosquito_class information like direction etc.
def find_mosquito_class(ground_truth_text):
    mosquito_class = ""
    my_list = ground_truth_text[0].split('\t')
    if my_list[0] is not None:
        mosquito_class = my_list[0]
    return mosquito_class

# getting mosquito_class name from predicted result
def extract_predicted_mosquito_class_name(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("name").get(0))
    return mosquito_class


# getting mosquito_class number from predicted result
def extract_predicted_mosquito_class_number(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("class").get(0))
    return mosquito_class


# getting mosquito_class confidence score from predicted result
def extract_predicted_mosquito_class_confidence(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("confidence").get(0))
    return mosquito_class

# getting mosquito bounding box from predicted result
def extract_predicted_mosquito_bbox(extractedInformation):
    bbox = []
    if extractedInformation is not None:
        xmin = str(extractedInformation.get("xmin").get(0))
        ymin = str(extractedInformation.get("ymin").get(0))
        xmax = str(extractedInformation.get("xmax").get(0))
        ymax = str(extractedInformation.get("ymax").get(0))
        bbox = [xmin, ymin, xmax, ymax]
    return bbox


# read information from a text file
# text files have been prepared manually
# by reading information from image
# and writing it to the text file
def read_mosquito_class(my_file):
    with open(my_file, "r", encoding="utf-8") as f:
        data = f.readlines()
        result = "".join(data)
    return result



# find image id
def find_image_id(original_image):
    image_name = os.path.splitext(original_image)[0]
    return image_name


# write row to previously created csv file
def write_row_to_csv(row):
    with open(my_csv_file, "w", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(row)


# path to upladed set of photos or to a single photo
root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/temp/"
#root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/self_made_photos/Monika/culex/"
#root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/self_made_photos/Mig/"

# path to transformed images
root_images = root_dataset + "images/"

# path mosquito_labels folder
# species_labels = root_dataset + "labels/"

all_images = os.listdir(root_images)
print(f"Total images: {len(all_images)}")

# counter for correctly recognized mosquito classes
counter = 0
labels_counter = 0
corr_percentage = 0

rows = []


# latest model trained on all classes
# different dataset split 90%/10%/0
#trained_model_pretrained = "/home/monika/Documents/models/exp27/best_exp27.pt"

# species only (model trained on dataset with six classes only, no blurred, no other species)
#trained_model_pretrained = "/home/monika/Documents/models/exp28/best_exp28.pt"

# yolov5 challenge trained baseline
trained_model_pretrained = "/home/monika/Documents/models/exp11_yolov5_baseline/best_exp11.pt"

# Load Custom Model

#yolov5_model  source='local' device='cpu' 
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path = trained_model_pretrained, force_reload = True)

# TODO check loading from local source
# yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path = trained_model_pretrained, force_reload=True, source='local')

                                



# after succesfull loading should have info like that:
# Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /home/monika/.cache/torch/hub/master.zip
# YOLOv5 🚀 2023-6-7 Python-3.10.6 torch-2.0.0+cu117 CPU

# Fusing layers... 
# Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
# Adding AutoShape... 

start = timer()
notFound = []

# manualy label mapping
# <key>: <value>,
# classes
# 0 aegypti
# 1 albopictus
# 2 culex
# 3 japonicus-koreicus 
# 4 other_species
# 5 xblurred
class_labels = {
    "aegypti":      0,
    "albopictus":   1,
    "anopheles":    2,
    "culex":        3,
    "culiseta":     4,
    "japonicus/koreicus": 5
}

for original_image in all_images:
    try:
        original_image_file = os.path.join(root_images, original_image)
        # checking if it is a file
        if os.path.isfile(original_image_file):
            # opening testing image
            print(f'You are watching: {original_image}')
            # classifying image by yolov5 model
            predictedInformation = classify_image(original_image_file)
            mosquito_class_name_predicted = ""
            mosquito_class_number_predicted = ""
            mosquito_class_confidence = ""
            mosquito_class_bbox = [0, 0, 0, 0]

            if predictedInformation:
                mosquito_class_name_predicted = extract_predicted_mosquito_class_name(predictedInformation)
                mosquito_class_number_predicted = extract_predicted_mosquito_class_number(predictedInformation)
                mosquito_class_confidence = extract_predicted_mosquito_class_confidence(predictedInformation)
                mosquito_class_bbox = extract_predicted_mosquito_bbox(predictedInformation)
                
            print(f"Predicted mosquito class: {mosquito_class_name_predicted} with {float(mosquito_class_confidence):.2f} confidence score.")
            #print(mosquito_class_name_predicted)
            #  bbox = [xmin, ymin, xmax, ymax]
            row = [original_image, mosquito_class_name_predicted, mosquito_class_number_predicted, mosquito_class_confidence, 
                   mosquito_class_bbox[0], mosquito_class_bbox[1], mosquito_class_bbox[2], mosquito_class_bbox[3]]
            rows.append(row)
            print(f"Finished file reading, file {original_image} read correctly!")
    except Exception as e:
        print(f'Unable to process file: {original_image}!')
        print(f'Exception: {e}!')
end = timer()
# counting how many percentage of mosquito_class have been recognized properly
if (counter):
    corr_percentage = counter / labels_counter * 100
# print results
print(f'Correctly recognized mosquitos: {counter} from {len(all_images)} images and from {labels_counter} labels, which gives us {corr_percentage:.2f}% of accuracy')

# create a csv file to write results of yolov5 classification
# checking if the file exists
csv_file_name = "results.csv"
destinationPath = root_dataset + "/" + csv_file_name
my_csv_file = Path(destinationPath)
if not my_csv_file.is_file():
    with open(destinationPath, 'w', encoding="utf-8") as f:
        f.close()

#header = ["image_name", "manually_read", "predicted_class", "similarity", "mosquito_class_recognized_corectly"]
header = ["image_name", "predicted_class_name", "predicted_class_number", "confidence_score", "xmin", "ymin", "xmax", "ymax"]
# create columns in csv files
if os.path.getsize(my_csv_file) == 0:
    with open(my_csv_file, "w", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        try:
            for row in rows:
                wr.writerow(row)
        except Exception as e:
            print(f'Unable to process row: {row}!')
            print(f'Exception: {e}!')

print(f'end: {end} - start: {start} = {end - start} (time in seconds)')
print(end - start)
print(notFound)