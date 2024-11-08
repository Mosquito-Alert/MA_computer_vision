# Script for calculate IoU and 
# F1 macro average score 
# and to check metrics and create confusion matrics
# Model trained for the AICrowd challenge
# baseline yolov5
# Install pip install scikit-learn
# pip install seaborn
# imports
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

# calculate Intersection over Union (IoU) for object detection
def intesection_over_union(gt_bbox, prd_bbox):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt_bbox[0], prd_bbox[0])
    yA = max(gt_bbox[1], prd_bbox[1])
    xB = min(gt_bbox[2], prd_bbox[2])
    yB = min(gt_bbox[3], prd_bbox[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
    boxBArea = (prd_bbox[2] - prd_bbox[0] + 1) * (prd_bbox[3] - prd_bbox[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# checks if mosquito class (manually classified)
# is present in image information
def check_mosquito_class(pred_mosquito_class, gt_mosquito_class):
    correct_prediction = False
    if gt_mosquito_class == pred_mosquito_class:
        correct_prediction = True
    return correct_prediction


# classes
class_labels = {
    "aegypti":              0,
    "albopictus":           1,
    "anopheles":            2,
    "culex":                3,
    "culiseta":             4,
    "japonicus/koreicus":   5, 
    "no-mosquito":          6
}

# path to gound truth csv
#csv_gt_root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/challenge_23v0.1-rc_baseline_full/public_test/"
csv_gt_root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/challenge_23v0.1-rc_baseline_full/private_test/"

# path to csv file with ground truth
#csv_gt_file_name = "public_test.csv"
csv_gt_file_name = "private_test.csv"

# path to predicted values csv
csv_pred_root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/challenge_23v0.1-rc_baseline_full/private_test/"

# path to csv file with predicted values
csv_pred_file_name = "baseline_predicted_results.csv"


# loading ground truth csv to dataframe
my_csv_gt_file_path = os.path.join(csv_gt_root_dataset, csv_gt_file_name)
my_csv_gt_file = Path(my_csv_gt_file_path)
if my_csv_gt_file.is_file():
    gt_results_df = pd.read_csv(my_csv_gt_file_path)

# loading predicted value csv to dataframe
my_csv_pred_file_path = os.path.join(csv_pred_root_dataset, csv_pred_file_name)
my_csv_pred_file = Path(my_csv_pred_file_path)
if my_csv_pred_file.is_file():
    pred_results_df = pd.read_csv(my_csv_pred_file_path)
    pred_results_df = pred_results_df.fillna("")


# counter for correctly calculated IoU
counterIoU = 0
# counter for correctly recognized mosquito classes
counter = 0
rows = []

# iterating through rows of predicted values csv file
for ind in pred_results_df.index:

    # original image name
    original_image_name = pred_results_df['image_name'][ind]

    print(f'You are watching: {original_image_name}')
    
    # prediction of image by trained model 
    pred_class_name = pred_results_df['predicted_class_name'][ind]

    # check if class name is empty
    if pred_class_name == '':
        pred_class_name = 'no-mosquito'

    # koreicus-japonicus
    if pred_class_name == 'japonicus-koreicus':
        pred_class_name = 'japonicus/koreicus'

    pred_class_number = pred_results_df['predicted_class_number'][ind]
    # check if class number is empty
    if pred_class_number == '':
        pred_class_number = '6'
    pred_confidence_score = pred_results_df['confidence_score'][ind]

    # predicted bbox
    pred_xmin = pred_results_df['xmin'][ind]
    pred_ymin = pred_results_df['ymin'][ind]
    pred_xmax = pred_results_df['xmax'][ind]
    pred_ymax = pred_results_df['ymax'][ind]
    pred_bbox = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]

    # find corresponding ground truth information
    ground_truth_row =  gt_results_df.loc[gt_results_df['img_fName'] == original_image_name]

    # ground truth labels
    gt_class_name = ground_truth_row['class_label'].values[0]
    gt_class_number = class_labels[gt_class_name]


    # ground truth bbox
    gt_xmin = float(ground_truth_row['bbx_xtl'].values[0])
    gt_ymin = float(ground_truth_row['bbx_ytl'].values[0])
    gt_xmax = float(ground_truth_row['bbx_xbr'].values[0])
    gt_ymax = float(ground_truth_row['bbx_ybr'].values[0])
    gt_bbox = [gt_xmin, gt_ymin, gt_xmax, gt_ymax]

    iou = intesection_over_union(gt_bbox, pred_bbox)
    print(iou)


    # original checking
    # flag for marking if mosquito class has been recognized correctly by trained model
    mosquito_class_recognized_corectly = 0
    correct_prediction = check_mosquito_class(pred_class_name, gt_class_name)
    if correct_prediction:
        counter += 1
        mosquito_class_recognized_corectly = 1

    # create row for csv file  
    row = [original_image_name, pred_class_name, pred_class_number, pred_confidence_score, \
                   pred_xmin, pred_ymin, pred_ymax, pred_ymax, \
                   gt_class_name, gt_class_number, gt_xmin, gt_ymin, gt_xmax, gt_ymax, \
                   mosquito_class_recognized_corectly, iou]
    
    rows.append(row)

# counting how many percentage of mosquito_class have been recognized properly
if (counter):
    rows_counter = len(gt_results_df.index)
    corr_percentage = counter / rows_counter * 100
# print results
print(f'Correctly recognized mosquitos: {counter}')
print(f'Correctly recognized mosquitos: {counter} from {(rows_counter)} images and from {rows_counter} labels, \
which gives us {corr_percentage:.2f}% of accuracy')

# create a csv file to write results of model classification
# checking if the file exists
csv_file_name = "results_challenge.csv"
destinationPath = csv_pred_root_dataset + "/" + csv_file_name
my_csv_file = Path(destinationPath)
if not my_csv_file.is_file():
    with open(destinationPath, 'w', encoding="utf-8") as f:
        f.close()

header = ["image_name", "pred_class_name", "pred_class_number", "pred_confidence_score", \
                   "pred_xmin", "pred_ymin", "pred_ymax", "pred_ymax", \
                   "gt_class_name", "gt_class_number", "gt_xmin", "gt_ymin", "gt_xmax", "gt_ymax", \
                   "mosquito_class_recognized_corectly", "iou"]

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
