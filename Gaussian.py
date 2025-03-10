import numpy as np
import argparse
import cv2
import os
import yaml

# Load class information from the data.yaml file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

class_names = data["names"]  # List of class names from the YAML file
class_map = {name: idx for idx, name in enumerate(class_names)}

print(f"Loaded class names from data.yaml: {class_names}")

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--radius", type=int, help="radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# Validate radius
if args["radius"] is None or args["radius"] <= 0 or args["radius"] % 2 == 0:
    print("Error: Invalid radius. It must be an odd number greater than 0. Using default value of 5.")
    args["radius"] = 5

image_folder = "dataset/train/converted_images"
label_folder = "dataset/train/gauss_labels"
bounded_image_folder = "dataset/train/bounded_images"

# Ensure folders exist
os.makedirs(label_folder, exist_ok=True)
os.makedirs(bounded_image_folder, exist_ok=True)

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Adaptive Intensity Thresholding
def adaptive_intensity_threshold(image, type='airglow'):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    if type == 'artifact':
        return mean_intensity + 2 * std_intensity  # Artifacts likely have higher intensity
    else:
        return mean_intensity + 1.5 * std_intensity  # Airglow has lower intensity


# Function to classify regions using both size and intensity
def classify_region(intensity, area, intensity_threshold, min_size=100, max_size=150):
    # Artifacts (e.g., the moon or stars) are likely to be small but have high intensity
    if intensity > intensity_threshold * 1.2 and area < min_size:  # High intensity and small area
        return 1  # Artifact
    elif area > max_size and intensity < intensity_threshold:
        return 0  # Airglow (larger and more diffuse)
    return 0  # Default to airglow


for image_path in image_paths:
    print(f"Processing image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
    
    # Compute adaptive intensity threshold
    INTENSITY_THRESHOLD = adaptive_intensity_threshold(blurred)
    print(f"Adaptive intensity threshold: {INTENSITY_THRESHOLD}")
    
    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    for _ in range(5):  # Detect top 5 bright regions
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blurred)
        x, y = maxLoc

        # Avoid overlapping detections
        while (x, y) in occupied_positions:
            x += step
            if x >= image.shape[1]:
                x = step
                y += step
            if y >= image.shape[0]:
                break
        
        area = step * step  # Approximate area of the region
        class_id = classify_region(maxVal, area, INTENSITY_THRESHOLD)
        bright_regions.append((x, y, class_id))
        occupied_positions.add((x, y))
        cv2.rectangle(blurred, (x - step, y - step), (x + step, y + step), 0, -1)
    
    label_file = os.path.join(label_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    
    with open(label_file, "w") as file:
        for x, y, class_id in bright_regions:
            norm_x_center = x / image.shape[1]
            norm_y_center = y / image.shape[0]
            norm_width = norm_height = args["radius"] * 2 / image.shape[1]
            
            file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")
            
            color = (255, 0, 0) if class_id == 1 else (0, 255, 0)
            cv2.rectangle(image, (x - args["radius"], y - args["radius"]), 
                          (x + args["radius"], y + args["radius"]), color, 2)
    
    output_image_path = os.path.join(bounded_image_folder, os.path.splitext(os.path.basename(image_path))[0] + "_processed.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {output_image_path}")

print("Processing complete.")