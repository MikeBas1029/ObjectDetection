from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'n' with 's', 'm', 'l', or 'x' for larger models

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on a new image
results = model.predict(source='path/to/test/image.jpg', save=True)

import cv2

# Load an image
img = cv2.imread('path/to/image.jpg')

# Iterate through YOLO results
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)  # Extract coordinates
    side = max(x2 - x1, y2 - y1)  # Find the longer side
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = center_x - side // 2, center_y - side // 2
    x2, y2 = center_x + side // 2, center_y + side // 2

    # Draw a square
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save or display the image
cv2.imwrite('output.jpg', img)