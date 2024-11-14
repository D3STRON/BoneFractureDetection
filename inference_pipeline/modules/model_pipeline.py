import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from ultralytics import YOLO
from modules import config
from modules import function
# from sklearn.metrics import precision_recall_curve, average_precision_score



def run_inference(models_to_run, mode):
    """Run inference and evaluation for specified models."""

    results = {}

    if mode == "localization":
        # Load COCO ground truth annotations
        coco_gt = COCO(config.annotation_file)
        imgIds = coco_gt.getImgIds()
        imgs = coco_gt.loadImgs(imgIds)
        filename_to_imgId = {img['file_name']: img['id'] for img in imgs}
        
        for model_name in models_to_run:
            predictions = []
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
                    img_filename = image_file.name
                    if img_filename not in filename_to_imgId:
                        print(f"Warning: Image {img_filename} not found in COCO annotations")
                        continue

                    img_id = filename_to_imgId[img_filename]
                    try:
                        result = model.predict(source=str(image_file), save=False, conf=0.5)[0]
                        for i in range(len(result.boxes.cls)):
                            class_id_model = int(result.boxes.cls[i].item())
                            
                            # Map model class IDs to COCO category IDs if necessary
                            # For example, if class IDs are off by 1
                            class_id = class_id_model + 1  # Adjust based on your dataset

                            conf = float(result.boxes.conf[i].item())
                            x_min, y_min, x_max, y_max = result.boxes.xyxy[i].tolist()
                            width = x_max - x_min
                            height = y_max - y_min
                            pred_dict = {
                                'image_id': img_id,
                                'category_id': class_id,
                                'bbox': [x_min, y_min, width, height],
                                'score': conf
                            }
                            predictions.append(pred_dict)
                    except Exception as e:
                        print(f"Error processing image {image_file}: {str(e)}")
                        continue
                
                print(f"Processed {len(predictions)} predictions successfully")

                # Evaluate predictions using COCOeval
                coco_dt = coco_gt.loadRes(predictions)
                cocoEval = COCOeval(coco_gt, coco_dt, iouType='bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                # Extract metrics
                metrics = {
                    'mAP': cocoEval.stats[0],       # mAP @ IoU=0.50:0.95
                    'mAP_50': cocoEval.stats[1],    # mAP @ IoU=0.50
                    'mAP_75': cocoEval.stats[2],    # mAP @ IoU=0.75
                    'AP_small': cocoEval.stats[3],
                    'AP_medium': cocoEval.stats[4],
                    'AP_large': cocoEval.stats[5],
                    'AR_1': cocoEval.stats[6],
                    'AR_10': cocoEval.stats[7],
                    'AR_100': cocoEval.stats[8],
                    'AR_small': cocoEval.stats[9],
                    'AR_medium': cocoEval.stats[10],
                    'AR_large': cocoEval.stats[11]
                }

                results[model_name] = {
                    'task_type': 'object_detection',
                    'predictions': predictions,
                    'metrics': metrics
                }

            elif model_name == "rcnn":
                # Placeholder for RCNN model
                print("Loading RCNN model...")
                # Implement RCNN inference similar to YOLO
                pass

    
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