import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# Load detection model
weight_file_path = "give/the/path/to/your/model"
model = YOLO(weight_file_path)

if model is None:
    print("Model failed to load")
else:
    print("Model loaded successfully")

# Directory paths
test_path = "give/the/path/to/test/set"
output_path = "give/the/path/to/save/the/results"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Function to run inference on a single image and save the result
def run_inference(image_path, output_path):
    # Run YOLOv8 inference
    results = model(image_path)

    # Read the original image
    image = cv2.imread(image_path)

    # Extract predictions from results and draw them on the image
    for result in results:
        boxes = result.boxes  # Access the bounding boxes
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get top-left and bottom-right points
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = result.names[class_id]  # Get class name

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the label with confidence score
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    save_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(save_path, image)

    print(f"Saved result for {image_path} to {save_path}")

# Iterate over images and run inference
def iterate_images(filenames, test_path):
    for file in filenames:
        image_path = os.path.join(test_path, file)
        run_inference(image_path, output_path)

# Get list of files in the test path
filenames = list(os.listdir(test_path))
iterate_images(filenames, test_path)
