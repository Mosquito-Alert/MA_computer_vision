import os
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

# Track the start time
start_time = time.time()

# Ensure the YOLOv8 model is available (pretrained on COCO)
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # yolov8n.pt is the smallest version of the YOLOv8 model

# Start training the model
print("Starting training...")
results = model.train(
    data='./dataset.yaml',  # Path to the dataset YAML file
    epochs=5,    # Reduced epochs due to small dataset size
    imgsz=320,   # Smaller image size to reduce memory load
    batch=8,     # Smaller batch size for laptop limitations
    name='yolov8_custom'  # Experiment name
)

print("Training completed. Starting evaluation...")

# Evaluate model on the test set and collect predictions
test_results = model.val(
    data='./dataset.yaml'  # Use the same YAML file for the test set
)

# Post-processing to calculate metrics
print("Evaluating predictions...")
predictions = test_results.pred
labels = test_results.labels

# Flattening lists for comparison
true_labels = [int(label[-1]) for label in labels]  # Last element is class
pred_labels = [int(pred[-1]) for pred in predictions]

# Calculate the necessary metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

# Calculate Intersection over Union (IoU) for bounding boxes
def iou(box1, box2):
    # Calculate intersection
    x1, y1, x2, y2 = np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[:2])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

ious = [iou(pred[:4], label[:4]) for pred, label in zip(predictions, labels)]
mean_iou = np.mean(ious)

# Save metrics to a plain text file
print("Saving metrics...")
with open('metrics_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Mean IoU: {mean_iou:.4f}\n")

# Track end time
end_time = time.time()
total_time = end_time - start_time
print(f"Process complete. Total running time: {total_time / 60:.2f} minutes.")
