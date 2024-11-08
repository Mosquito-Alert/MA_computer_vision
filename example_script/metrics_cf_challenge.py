# Script for calculate IoU and 
# F1 macro average score 
# and to check metrics and create confusion matrics
# Model trained for the AICrowd challenge
# baseline yolov5

# imports
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


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

# path to result csv
csv_results_root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/challenge_23v0.1-rc_baseline_full/private_test/"
#csv_results_root_dataset = "/home/monika/datasets_not_sync/datasets_for_manipulations/challenge_yolov5/public_test/"

# path to csv file with results
#csv_results_file_name = "results_challenge.csv"
csv_results_file_name = "results_challenge.csv"


# loading results csv to dataframe
my_csv_file_path = os.path.join(csv_results_root_dataset, csv_results_file_name)
my_csv_file = Path(my_csv_file_path)
if my_csv_file.is_file():
    results_df = pd.read_csv(my_csv_file_path)
    results_df = results_df.fillna(0)


# Metrics
# ground truth (actual value)
actualValue = results_df['gt_class_number'].values # y_test

# prediction of image by trained model 
predictedValue = results_df['pred_class_number'].values # y_pred

print(f'\nMetrics:')

# IoU distribution
iouValues = results_df['iou'].values 
mean_iou = iouValues.mean()
print(f'Mean IoU:                       {mean_iou:.3f}')



# accuracy
acc_score = accuracy_score(actualValue, predictedValue)
print(f'Accuracy:                       {acc_score:.3f}')

# Average - none
print(f"\nNone averaged:")
# None-averaged precission
none_precision = precision_score(actualValue, predictedValue, average=None, zero_division=0) 
print(f'Precision (average none):       {none_precision}')

# None-averaged recall
none_recall = recall_score(actualValue, predictedValue, average=None, zero_division=0)
print(f'Recall (average none):          {none_recall}')

# None-averaged F1 score
none_f1score = f1_score(actualValue, predictedValue, average=None, zero_division=0)
print(f'F1 score (average none):        {none_f1score}')

# Macro
print(f"\nMacro averaged:")
# Macro-averaged precission
macro_precision = precision_score(actualValue, predictedValue, average='macro', zero_division=0)
print(f'Macro-averaged precision:       {macro_precision:.3f}')

# Macro-averaged recall
macro_recall = recall_score(actualValue, predictedValue, average='macro', zero_division=0)
print(f'Macro-averaged recall:          {macro_recall:.3f}')

# Macro-averaged F1 score
macro_f1score = f1_score(actualValue, predictedValue, average='macro', zero_division=0)
print(f'Macro-averaged F1 score:        {macro_f1score:.3f}')

# Micro
print(f"\nMicro averaged:")
# Micro-averaged precission
micro_precision = precision_score(actualValue, predictedValue, average='micro')
print(f'Micro-averaged precision:       {micro_precision:.3f}')

# Micro-averaged recall
micro_recall = recall_score(actualValue, predictedValue, average='micro')
print(f'Micro-averaged recall:          {micro_recall:.3f}')

# Micro-averaged F1 score
micro_f1score = f1_score(actualValue, predictedValue, average='micro')
print(f'Micro-averaged F1 score:        {micro_f1score:.3f}')

# Weighted-averaged
print(f"\nWeighted-averaged:")
# Weighted precission
weighted_precision = precision_score(actualValue, predictedValue, average='weighted', zero_division=0)
print(f'Weighted-averaged precision:    {weighted_precision:.3f}')

# Weighted recall
weighted_recall = recall_score(actualValue, predictedValue, average='weighted', zero_division=0)
print(f'Weighted-averaged recall:       {weighted_recall:.3f}')

# Weighted F1 score
weighted_f1score = f1_score(actualValue, predictedValue, average='weighted', zero_division=0)
print(f'Weighted-averaged F1 score:     {weighted_f1score:.3f}')

# print classification report for model
display_labels = ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus/koreicus', 'no-mosquito']
print("\nClassification_report:")
print(classification_report(actualValue, predictedValue, target_names=display_labels, zero_division=0))

# confusion matrix
cls_labels = [0, 1, 2, 3, 4, 5, 6]
conf_matrix = confusion_matrix(actualValue, predictedValue, labels=cls_labels) #, labels = cls_labels

#create seaborn displot
ax = sns.displot(data = iouValues, color='steelblue',  height=3, aspect=2)

# Change figure size and increase dpi for better resolution
#plt.figure(figsize=(20,20))
ax.fig.set_dpi(200)

# specify axis labels
# plt.xlabel('IoU', size=10)
# plt.ylabel('Samples', size=10)
# plt.title('The Intersection-over-Union (IoU) distributions of the baseline (yolov5)', size = 10)
ax.set(xlabel='IoU',
       ylabel='Samples',
       title='The Intersection-over-Union (IoU) distributions of the baseline (yolov5)')

# display displot
plt.show()



#Scale up the size of all text
plt.figure(figsize=(20,20))
sns.set(font_scale = 0.7)

print("conf_matrix:")
print(conf_matrix)
# sns.heatmap(conf_matrix, annot=True, cmap='Blues')
# plt.show()

# Plot Confusion Matrix using Seaborn heatmap()
# Parameters:
# first param - confusion matrix in array format   
# annot = True: show the numbers in each heatmap cell
# fmt = 'd': show numbers as integers. 
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

# set x-axis label and ticks. 
ax.set_xlabel("Predicted class", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(display_labels)

# set y-axis label and ticks
ax.set_ylabel("Actual class", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(display_labels)

# set plot title
ax.set_title("Confusion Matrix for the yolov5 detection and classification model", fontsize=14, pad=20)

plt.show()

print(f'Macro-averaged F1 score:        {macro_f1score}')
print(f'Mean IoU:                       {mean_iou}')