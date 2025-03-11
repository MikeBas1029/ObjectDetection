import os
import cv2
from ultralytics import YOLO
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from data_loader import data_loader

# cmd for boxes labelImg dataset/train/converted_images (must in each folder have classes.txt)

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'n' with 's', 'm', 'l', or 'x' for larger models

# Train the model
model.train(data='data.yaml', epochs=20, imgsz=640, batch=16)

# Load the trained model
model = YOLO('runs/detect/train6/weights/best.pt')

# Predict on a new image
results = model.predict(source='dataset/train/converted_images/PKR_DASC_0428_20160217_143735.846.png', save=True)

# Load an image
img = cv2.imread('dataset/train/converted_images/PKR_DASC_0428_20160217_143735.846.png')

# Iterate through YOLO results
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
    print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}")
    side = max(x2 - x1, y2 - y1)  # Find the longer side
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = center_x - side // 2, center_y - side // 2
    x2, y2 = center_x + side // 2, center_y + side // 2

    # Draw a square
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save or display the image
cv2.imwrite('output.jpg', img)
