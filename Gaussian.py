import numpy as np
import argparse
import cv2
import os
import yaml

# Load class information from the data.yaml file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

class_names = data["names"]  # This is the list of class names from the YAML file
class_map = {name: idx for idx, name in enumerate(class_names)}

# Print the loaded class names from the YAML file
print(f"Loaded class names from data.yaml: {class_names}")

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--radius", type=int, help="radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# Check if the radius is valid
if args["radius"] is None or args["radius"] <= 0 or args["radius"] % 2 == 0:
    print("Error: Invalid radius. It must be an odd number greater than 0. Using default value of 5.")
    args["radius"] = 5  # Set default radius

image_folder = "dataset/train/converted_images"
#label_folder = "dataset/train/gauss_labels"  # Folder to save label files
#bounded_image_folder = "dataset/train/bounded_images"  # Folder to save processed images

# Ensure that the folders exist
if not os.path.exists(image_folder):
    print(f"Error: Folder {image_folder} does not exist.")
    exit()

#if not os.path.exists(label_folder):
#    os.makedirs(label_folder)

#if not os.path.exists(bounded_image_folder):
#    os.makedirs(bounded_image_folder)

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image in the folder
for image_path in image_paths:
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    for _ in range(5):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        x, y = maxLoc
        
        # Adjust position to avoid overlap but keep boxes connected
        while (x, y) in occupied_positions:
            x += step
            if x >= image.shape[1]:
                x = step
                y += step
            if y >= image.shape[0]:
                break
        
        bright_regions.append((x, y))
        occupied_positions.add((x, y))
        cv2.rectangle(gray, (x - step, y - step), (x + step, y + step), 0, -1)  # Mask out region

    for region in bright_regions:
        top_left = (region[0] - args["radius"], region[1] - args["radius"])
        bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

    image_height, image_width, _ = image.shape
    label_file = os.path.join(image_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    
    with open(label_file, "w") as file:
        for idx, region in enumerate(bright_regions):
            class_id = 0  # Assign class 0 for bright regions
            x_center, y_center = region
            width = height = args["radius"] * 2
            norm_x_center = (x_center / image_width)
            norm_y_center = (y_center / image_height)
            norm_width = (width / image_width)
            norm_height = (height / image_height)
            file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")
    
    #output_image_path = os.path.join(bounded_image_folder, os.path.splitext(os.path.basename(image_path))[0] + "_processed.jpg")
    #cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {image_folder}")