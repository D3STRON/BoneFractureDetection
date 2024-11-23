from modules import model_pipeline
from modules import config
import os

if __name__ == "__main__":
    # Verify paths exist
    if not os.path.exists(config.images_path):
        print(f"Error: Images path {config.images_path} does not exist")
        exit(1)
    if not os.path.exists(config.annotation_file):
        print(f"Error: Labels path {config.annotation_file} does not exist")
        exit(1)
        
    # Specify the models you want to run 
    # NOTE: pipeline is configured to run localization tasks only or classification tasks only not both.
    models_to_run = ['detr']

    # specify mode
    mode = config.task
    
    # Run inference
    results = model_pipeline.run_inference(models_to_run, mode)
    
    if results:
        # Print detailed results
        for model_name, model_results in results.items():
            task_type = model_results['task_type']
            metrics = model_results['metrics']
            
            print(f"\nModel: {model_name}")
            if task_type == "object_detection":
                print("Task Type: Object Detection")
                print(f"mAP@IoU=0.50:0.95: {metrics['mAP']:.4f}")
                print(f"mAP@IoU=0.50: {metrics['mAP_50']:.4f}")
                print(f"mAP@IoU=0.75: {metrics['mAP_75']:.4f}")
                print(f"AP (Small): {metrics['AP_small']:.4f}")
                print(f"AP (Medium): {metrics['AP_medium']:.4f}")
                print(f"AP (Large): {metrics['AP_large']:.4f}")
                print(f"AR@1: {metrics['AR_1']:.4f}")
                print(f"AR@10: {metrics['AR_10']:.4f}")
                print(f"AR@100: {metrics['AR_100']:.4f}")
                print(f"AR (Small): {metrics['AR_small']:.4f}")
                print(f"AR (Medium): {metrics['AR_medium']:.4f}")
                print(f"AR (Large): {metrics['AR_large']:.4f}")
            elif task_type == "classification":
                print("Task Type: Classification")
                print(f"Accuracy: {metrics['accuracy']}")
                print(f"Precision: {metrics['precision']}")
                print(f"Recall: {metrics['recall']}")
                print(f"F1 Score: {metrics['f1_score']}")
                print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
            