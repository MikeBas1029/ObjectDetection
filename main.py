import os
import cv2
from ultralytics import YOLO
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# cmd for boxes labelImg dataset/train/converted_images (must in each folder have classes.txt)
def data_loader(input_dir, output_dir):
    """
    Converts FITS files in the input directory to PNG files in the output directory.
    
    :param input_dir: Path to the directory containing FITS files.
    :param output_dir: Path to the directory where converted PNG files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Iterate over all FITS files in the input directory
    for fits_file in os.listdir(input_dir):
        if fits_file.endswith(".FITS"):
            output_file = fits_file.replace(".FITS", ".png")  # Corresponding PNG file name
            output_path = os.path.join(output_dir, output_file)
            
            # Check if the PNG file already exists
            if os.path.exists(output_path):
                print(f"Skipping {fits_file}, {output_file} already exists.")
                continue
            
            fits_path = os.path.join(input_dir, fits_file)
            try:
                # Open FITS file and load data
                with fits.open(fits_path) as hdul:
                    data = hdul[0].data.astype(np.float32)  # Convert to float32 for processing

                # Normalize the data for visualization
                data = np.nan_to_num(data)  # Replace NaNs with zeros
                data -= data.min()  # Shift to zero
                if data.max() > 0:  # Avoid division by zero
                    data /= data.max()  # Normalize to [0, 1]
                data *= 255  # Scale to [0, 255]

                # Save as PNG
                plt.imsave(output_path, data, cmap="gray")
                print(f"Converted {fits_file} to {output_path}")
            except Exception as e:
                print(f"Error converting {fits_file}: {e}")

# Define paths relative to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
train_fits_path = os.path.join(project_root, "dataset/train/images")
train_png_path = os.path.join(project_root, "dataset/train/converted_images")
train_labels_path = os.path.join(project_root, "dataset/train/labels")
val_fits_path = os.path.join(project_root, "dataset/val/images")
val_png_path = os.path.join(project_root, "dataset/val/converted_images")
val_labels_path = os.path.join(project_root, "dataset/val/labels")

# Convert FITS files to PNG for both training and validation datasets
data_loader(train_fits_path, train_png_path)
data_loader(val_fits_path, val_png_path)

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
