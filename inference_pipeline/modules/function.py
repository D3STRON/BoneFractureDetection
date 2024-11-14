import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def pre_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224 
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def load_data(dataset_path):
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=pre_process())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader



# def convert_yolo_to_corners(box, img_width, img_height):
#     """
#     Convert YOLO format (class_id, x_center, y_center, width, height) to corner coordinates
#     (class_id, x_min, y_min, x_max, y_max).
#     """
#     if len(box) < 5:
#         raise ValueError(f"Invalid box format: {box}")
        
#     class_id = box[0]
#     x_center, y_center = box[1], box[2]
#     width, height = box[3], box[4]
    
#     # Convert normalized coordinates to absolute coordinates
#     x_min = (x_center - width/2) * img_width
#     y_min = (y_center - height/2) * img_height
#     x_max = (x_center + width/2) * img_width
#     y_max = (y_center + height/2) * img_height
    
#     return [class_id, x_min, y_min, x_max, y_max]

# def compute_iou(box1, box2):
#     """
#     Compute Intersection over Union (IoU) between two boxes in corner format
#     (class_id, x_min, y_min, x_max, y_max).
#     """
#     if len(box1) < 5 or len(box2) < 5:
#         raise ValueError(f"Invalid box format: box1={box1}, box2={box2}")
    
#     # Extract coordinates (ignore class_id at index 0)
#     box1_x1, box1_y1, box1_x2, box1_y2 = box1[1:5]
#     box2_x1, box2_y1, box2_x2, box2_y2 = box2[1:5]
    
#     # Calculate intersection coordinates
#     x_left = max(box1_x1, box2_x1)
#     y_top = max(box1_y1, box2_y1)
#     x_right = min(box1_x2, box2_x2)
#     y_bottom = min(box1_y2, box2_y2)
    
#     # Check if there is no intersection
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
    
#     # Calculate intersection area
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
#     # Calculate union area
#     box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
#     box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
#     union_area = box1_area + box2_area - intersection_area
    
#     # Handle edge case where union area is zero
#     if union_area <= 0:
#         return 0.0
        
#     # Calculate IoU
#     iou = intersection_area / union_area
    
#     return iou

# def load_ground_truth_boxes(label_path):
#     """Load ground truth bounding boxes from label files in YOLO format."""
#     ground_truths = {}
#     if not os.path.exists(label_path):
#         print(f"Warning: Label path {label_path} does not exist")
#         return ground_truths
        
#     for filename in os.listdir(label_path):
#         if not filename.endswith('.txt'):
#             continue
            
#         file_path = os.path.join(label_path, filename)
#         with open(file_path, 'r') as file:
#             boxes = []
#             for line in file:
#                 try:
#                     class_id, x_center, y_center, width, height = map(float, line.strip().split())
#                     boxes.append([class_id, x_center, y_center, width, height])
#                 except ValueError as e:
#                     print(f"Warning: Skipping malformed line in {filename}: {line.strip()}")
#                     continue
#             ground_truths[filename.split('.')[0]] = boxes
    
#     if not ground_truths:
#         print("Warning: No valid ground truth boxes loaded")
#     else:
#         print(f"Loaded ground truth boxes for {len(ground_truths)} images")
#     return ground_truths

# def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):
#     """Evaluate precision, recall, and mAP for predicted boxes."""
#     if not predictions or not ground_truths:
#         print("Error: Empty predictions or ground truths")
#         return None, None, 0.0

#     true_positives = []
#     false_positives = []
#     scores = []
#     n_ground_truths = 0

#     # Process each image
#     for img_id, pred_boxes in predictions.items():
#         if img_id not in ground_truths:
#             print(f"Warning: No ground truth found for image {img_id}")
#             continue

#         gt_boxes = ground_truths[img_id]
#         n_ground_truths += len(gt_boxes)
        
#         if not pred_boxes:
#             continue

#         # Convert ground truth boxes to corner format for IoU calculation
#         try:
#             gt_boxes_corners = [convert_yolo_to_corners(box, 640, 640) for box in gt_boxes]  # Adjust size if needed
#         except ValueError as e:
#             print(f"Error converting ground truth boxes for image {img_id}: {e}")
#             continue
        
#         # Track which ground truth boxes have been matched
#         gt_matched = [False] * len(gt_boxes_corners)

#         # Sort predictions by confidence
#         pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)

#         for pred_box in pred_boxes:
#             scores.append(pred_box[5])
            
#             max_iou = 0
#             max_idx = -1

#             # Find the best matching ground truth box
#             for j, gt_box in enumerate(gt_boxes_corners):
#                 if not gt_matched[j] and pred_box[0] == gt_box[0]:  # Same class
#                     try:
#                         iou = compute_iou(pred_box, gt_box)
#                         if iou > max_iou:
#                             max_iou = iou
#                             max_idx = j
#                     except ValueError as e:
#                         print(f"Error computing IoU: {e}")
#                         continue

#             if max_iou >= iou_threshold:
#                 true_positives.append(1)
#                 false_positives.append(0)
#                 gt_matched[max_idx] = True
#             else:
#                 true_positives.append(0)
#                 false_positives.append(1)

#     if not scores:
#         print("No valid predictions for evaluation")
#         return None, None, 0.0

#     # Convert to numpy arrays
#     scores = np.array(scores)
#     true_positives = np.array(true_positives)
#     false_positives = np.array(false_positives)

#     # Sort by confidence
#     sorted_indices = np.argsort(-scores)
#     true_positives = true_positives[sorted_indices]
#     false_positives = false_positives[sorted_indices]

#     # Compute cumulative sums
#     true_positives_cumsum = np.cumsum(true_positives)
#     false_positives_cumsum = np.cumsum(false_positives)

#     # Compute precision and recall
#     precision = true_positives_cumsum / (true_positives_cumsum + false_positives_cumsum)
#     recall = true_positives_cumsum / n_ground_truths if n_ground_truths > 0 else np.zeros_like(true_positives_cumsum)

#     # Compute AP
#     ap = 0
#     for i in range(len(precision)-1):
#         ap += (recall[i+1] - recall[i]) * precision[i+1]

#     return precision, recall, ap

def evaluate_predictions(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy, precision, recall, F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, conf_matrix