import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from pathlib import Path
from ultralytics import YOLO
from modules import config
from modules import function
# from sklearn.metrics import precision_recall_curve, average_precision_score



def run_inference(models_to_run, mode):
    """Run inference and evaluation for specified models."""

    results = {}

    if mode == "localization":
        # Load ground truth boxes
        ground_truth_boxes = function.load_ground_truth_boxes(config.label_path)
        if not ground_truth_boxes:
            print("Error: No ground truth boxes loaded")
            return None
    
        for model_name in models_to_run:
            predictions = {}
            print(f"\nRunning inference for {model_name}...")
            
            if model_name == "yolo":
                model = YOLO(config.yolo_path)
                
                # Check if images path exists
                if not os.path.exists(config.images_path):
                    print(f"Error: Images path {config.images_path} does not exist")
                    continue
                    
                # Run inference on validation images
                image_files = list(Path(config.images_path).glob('*.jpg'))
                if not image_files:
                    print(f"Error: No jpg images found in {config.images_path}")
                    continue
                    
                print(f"Processing {len(image_files)} images...")
                
                for image_file in image_files:
                    img_id = image_file.stem
                    try:
                        result = model.predict(source=str(image_file), save=False, conf=0.5)[0]
                        pred_boxes = []
                        
                        for i in range(len(result.boxes.cls)):
                            class_id = int(result.boxes.cls[i].item())
                            conf = float(result.boxes.conf[i].item())
                            x_min, y_min, x_max, y_max = result.boxes.xyxy[i].tolist()
                            pred_boxes.append([class_id, x_min, y_min, x_max, y_max, conf])
                        
                        predictions[img_id] = pred_boxes
                        
                    except Exception as e:
                        print(f"Error processing image {image_file}: {str(e)}")
                        continue
                
                print(f"Processed {len(predictions)} images successfully")
                
                # Evaluate predictions
                precision, recall, ap = function.evaluate_detections(predictions, ground_truth_boxes)
                if precision is not None and recall is not None:
                    avg_precision = np.mean(precision)
                    avg_recall = np.mean(recall)
                    print(f"\n{model_name.upper()} Evaluation Results:")
                    print(f"Average Precision: {avg_precision:.4f}")
                    print(f"Average Recall: {avg_recall:.4f}")
                    print(f"mAP: {ap:.4f}")
                    
                    results[model_name] = {
                    'task_type': 'object_detection',
                    'predictions': predictions,
                    'metrics': {
                        'average_precision': np.mean(precision) if precision is not None else None,
                        'average_recall': np.mean(recall) if recall is not None else None,
                        'mAP': ap if ap is not None else None,
                        'accuracy': None,
                        'precision': None,
                        'recall': None,
                        'f1_score': None,
                        'confusion_matrix': None
                    }
                }
                else:
                    print(f"Evaluation failed for {model_name}")

            elif model_name == "rcnn":
                # Placeholder for RCNN model loading and inference
                print("Loading RCNN model...")
                model = None  # Placeholder for actual RCNN model loading
                
                # Placeholder check for image path existence
                if not os.path.exists(config.images_path):
                    print(f"Error: Images path {config.images_path} does not exist")
                    continue
                
                # Run inference on validation images (Placeholder)
                image_files = list(Path(config.images_path).glob('*.jpg'))
                if not image_files:
                    print(f"Error: No jpg images found in {config.images_path}")
                    continue

                print(f"Processing {len(image_files)} images with RCNN (placeholder)...")
                
                # Placeholder for RCNN inference (populate `pred_boxes` similar to YOLO format)
                for image_file in image_files:
                    img_id = image_file.stem
                    try:
                        # Placeholder inference result
                        pred_boxes = [
                            # Example box format: [class_id, x_min, y_min, x_max, y_max, conf]
                            [0, 100, 150, 200, 250, 0.9]  # Example prediction
                        ]
                        predictions[img_id] = pred_boxes
                        
                    except Exception as e:
                        print(f"Error processing image {image_file}: {str(e)}")
                        continue
                
                print(f"Processed {len(predictions)} images successfully with RCNN placeholder")

                precision, recall, ap = function.evaluate_detections(predictions, ground_truth_boxes)
                if precision is not None and recall is not None:
                    avg_precision = np.mean(precision)
                    avg_recall = np.mean(recall)
                    print(f"\n{model_name.upper()} Evaluation Results:")
                    print(f"Average Precision: {avg_precision:.4f}")
                    print(f"Average Recall: {avg_recall:.4f}")
                    print(f"mAP: {ap:.4f}")
                    
                    results[model_name] = {
                    'task_type': 'object_detection',
                    'predictions': predictions,
                    'metrics': {
                        'average_precision': np.mean(precision) if precision is not None else None,
                        'average_recall': np.mean(recall) if recall is not None else None,
                        'mAP': ap if ap is not None else None,
                        'accuracy': None,
                        'precision': None,
                        'recall': None,
                        'f1_score': None,
                        'confusion_matrix': None
                    }
                }
                else:
                    print(f"Evaluation failed for {model_name}")

    
    else:
        
        for model_name in models_to_run:
            if model_name == "resnet":
                print("Loading ResNet model...")
                resnet_model_path = config.resnet_path
                model = models.resnet18(pretrained=False)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 2)  # Binary classification output
                model.load_state_dict(torch.load(resnet_model_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)

                # load the data for classification evaluation

                test_loader = function.load_data(config.classification_test_data)

                if test_loader is None:
                    print("Error: Test loader for ResNet model is not provided")
                    continue

                # Evaluate the ResNet model on the classification task
                accuracy, precision, recall, f1, conf_matrix = function.evaluate_predictions(model, test_loader, device)
                print(f"\n{model_name.upper()} Evaluation Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"Confusion Matrix:\n{conf_matrix}")

                results[model_name] = {
                    'task_type': 'classification',
                    'predictions': None,
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confusion_matrix': conf_matrix,
                        'average_precision': None,
                        'average_recall': None,
                        'mAP': None
                    }
                }
            elif model_name == "vit":
                print("loading vit model...")
                # place the model loading part for vit and send it to evaluation_predictions
                results[model_name] = {
                    'task_type': 'classification',
                    'predictions': None,
                    'metrics': {
                        'accuracy': 0.85,  # Placeholder values
                        'precision': 0.8,
                        'recall': 0.82,
                        'f1_score': 0.81,
                        'confusion_matrix': [[10, 2], [3, 15]],
                        'average_precision': None,
                        'average_recall': None,
                        'mAP': None
                    }
                }

    return results