'''
import numpy as np
import argparse
import cv2
import os
import yaml
from scipy.stats import norm

# Load class information from the data.yaml file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

class_names = data["names"]  # List of class names from YAML file
class_map = {name: idx for idx, name in enumerate(class_names)}

print(f"Loaded class names from data.yaml: {class_names}")

# Argument parser for Gaussian blur radius
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--radius", type=int, help="Radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# Check if the radius is valid
if args["radius"] is None or args["radius"] <= 0 or args["radius"] % 2 == 0:
    print("Error: Invalid radius. It must be an odd number greater than 0. Using default value of 5.")
    args["radius"] = 5  # Default radius

# Folder paths
image_folder = "dataset/train/converted_images"
bounded_image_folder = "dataset/train/bounded_images"
false_origin_folder = "dataset/train/false_origins"

# Ensure necessary folders exist
os.makedirs(false_origin_folder, exist_ok=True)
os.makedirs(bounded_image_folder, exist_ok=True)

# Get image paths
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    # Initialize contours (in case no stars are detected)
    contours = []

    # Compute the average brightness of the image
    avg_brightness = np.mean(gray)

    # Airglow Detection (only if the image is bright enough)
    if avg_brightness > 10:  # Adjust this threshold if needed
        for _ in range(5):
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred_gray)
            x, y = maxLoc

            # Ensure no duplicate regions
            while (x, y) in occupied_positions:
                x += step
                if x >= image.shape[1]:
                    x = step
                    y += step
                if y >= image.shape[0]:
                    break

            if maxVal > avg_brightness * 1.2:  # Only detect regions significantly brighter than the background
                bright_regions.append((x, y))
                occupied_positions.add((x, y))
                cv2.rectangle(blurred_gray, (x - step, y - step), (x + step, y + step), 0, -1)  # Mask out detected region

    # Draw bounding boxes for airglow (only if bright regions were found)
    if bright_regions:
        for region in bright_regions:
            top_left = (region[0] - args["radius"], region[1] - args["radius"])
            bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue for airglow

    # Detect stars (only if the image has sufficient contrast)
    if avg_brightness > 2:  # Adjust this threshold if needed
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean, std = norm.fit(thresholded)
        star_mask = np.abs(thresholded - mean) < 2 * std

        contours, _ = cv2.findContours(star_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes for stars (using a green color)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 and h < 10:  # Stars should be small
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for stars

    output_image_path = os.path.join(bounded_image_folder, os.path.splitext(os.path.basename(image_path))[0] + "_processed.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {output_image_path}")
print("Processing complete!")
'''
import numpy as np
import argparse
import cv2
import os
import yaml
from scipy.stats import norm

# Load class information from the data.yaml file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

class_names = data["names"]  # List of class names from YAML file
class_map = {name: idx for idx, name in enumerate(class_names)}

print(f"Loaded class names from data.yaml: {class_names}")

# Argument parser for Gaussian blur radius
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--radius", type=int, help="Radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# Check if the radius is valid
if args["radius"] is None or args["radius"] <= 0 or args["radius"] % 2 == 0:
    print("Error: Invalid radius. It must be an odd number greater than 0. Using default value of 5.")
    args["radius"] = 5  # Default radius

# Folder paths
image_folder = "dataset/train/converted_images"
bounded_image_folder = "dataset/train/bounded_images"
false_origin_folder = "dataset/train/false_origins"

# Ensure necessary folders exist
os.makedirs(false_origin_folder, exist_ok=True)
os.makedirs(bounded_image_folder, exist_ok=True)

# Get image paths
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    # Initialize contours (in case no stars are detected)
    contours = []

    # Compute the average brightness of the image
    avg_brightness = np.mean(gray)

    # Airglow Detection (only if the image is bright enough)
    if avg_brightness > 10:  # Adjust this threshold if needed
        for _ in range(5):
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred_gray)
            x, y = maxLoc

            # Ensure no duplicate regions
            while (x, y) in occupied_positions:
                x += step
                if x >= image.shape[1]:
                    x = step
                    y += step
                if y >= image.shape[0]:
                    break

            if maxVal > avg_brightness * 1.2:  # Only detect regions significantly brighter than the background
                bright_regions.append((x, y))
                occupied_positions.add((x, y))
                cv2.rectangle(blurred_gray, (x - step, y - step), (x + step, y + step), 0, -1)  # Mask out detected region

    # Draw bounding boxes for airglow (only if bright regions were found)
    if bright_regions:
        for region in bright_regions:
            top_left = (region[0] - args["radius"], region[1] - args["radius"])
            bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue for airglow

    # Detect stars (only if the image has sufficient contrast)
    if avg_brightness > 2:  # Adjust this threshold if needed
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean, std = norm.fit(thresholded)
        star_mask = np.abs(thresholded - mean) < 2 * std

        contours, _ = cv2.findContours(star_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes for stars (using a green color)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 300 and h < 300:  # Stars should be small
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for stars

    output_image_path = os.path.join(bounded_image_folder, os.path.splitext(os.path.basename(image_path))[0] + "_processed.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {output_image_path}")
print("Processing complete!")