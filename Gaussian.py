'''
import numpy as np
import argparse
import cv2
import os
import yaml

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

# Function to check if two rectangles overlap
def is_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1:  # No horizontal overlap
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:  # No vertical overlap
        return False
    return True

# Loop through each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions and define circular mask
    height, width = gray.shape
    center = (width // 2, height // 2)
    radius = min(center)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply Gaussian Blur before detection
    blurred_gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    # Compute the average brightness of the image
    avg_brightness = np.mean(gray[mask == 255])  # Only consider inside circular mask

    # List to store airglow bounding boxes for overlap check
    airglow_bboxes = []

    # -----------------------------
    #  AIRGLOW DETECTION (BLUE BOXES)
    # -----------------------------
    if avg_brightness > 10:
        for _ in range(5):  # Detect up to 5 bright regions
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

            # Only detect if significantly brighter than average
            if maxVal > avg_brightness * 1.2:
                bright_regions.append((x, y))
                occupied_positions.add((x, y))
                cv2.rectangle(blurred_gray, (x - step, y - step), (x + step, y + step), 0, -1)  # Mask detected region

    # Draw bounding boxes for airglow and save the coordinates
    for region in bright_regions:
        top_left = (region[0] - args["radius"], region[1] - args["radius"])
        bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
        airglow_bboxes.append((top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue for airglow

    # -----------------------------
    #  STAR DETECTION (GREEN BOXES)
    # -----------------------------
    gray_no_blur = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)  # Original grayscale image
    laplacian = cv2.Laplacian(gray_no_blur, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)

    # Apply circular mask
    masked_sharpness = cv2.bitwise_and(sharpness_map, mask)

    # Increase threshold for better star detection
    star_threshold = avg_brightness * 3.8
    detected_stars = []

    # Detect bright sharp spots (stars)
    for _ in range(5):  # Detect up to 5 bright spots
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(masked_sharpness)
        x, y = maxLoc

        # Ensure detection is inside the circular region
        if (x - center[0])**2 + (y - center[1])**2 < radius**2:
            if maxVal > star_threshold:
                bbox_size = args["radius"] // 3  # Smaller boxes for stars
                top_left = (x - bbox_size, y - bbox_size)
                bottom_right = (x + bbox_size, y + bbox_size)

                # Check if the star bounding box overlaps with any airglow box
                overlap_found = False
                for airglow_bbox in airglow_bboxes:
                    if is_overlap((top_left[0], top_left[1], bbox_size*2, bbox_size*2), airglow_bbox):
                        overlap_found = True
                        break

                # Only draw green bounding box if no overlap
                if not overlap_found:
                    detected_stars.append((x, y))
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green for stars

                    # Mask out detected star to prevent repeated detection
                    cv2.rectangle(masked_sharpness, top_left, bottom_right, 0, -1)

    # -----------------------------
    #  SAVE PROCESSED IMAGE
    # -----------------------------
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

# Function to check if two rectangles overlap
def is_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1:  # No horizontal overlap
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:  # No vertical overlap
        return False
    return True

# Loop through each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions and define circular mask
    height, width = gray.shape
    center = (width // 2, height // 2)
    radius = min(center)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply Gaussian Blur before detection
    blurred_gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2

    # Compute the average brightness of the image
    avg_brightness = np.mean(gray[mask == 255])  # Only consider inside circular mask

    # List to store airglow bounding boxes for overlap check
    airglow_bboxes = []
    
    # Define a threshold for ignoring detections at the top
    top_threshold = int(0.15 * height)  # Ignore top 15% of the image

    # -----------------------------
    #  AIRGLOW DETECTION (BLUE BOXES)
    # -----------------------------
    if avg_brightness > 10:
        for _ in range(5):  # Detect up to 5 bright regions
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

            # Only detect if significantly brighter than average
            if maxVal > avg_brightness * 1.2 and y > top_threshold:
                bright_regions.append((x, y))
                occupied_positions.add((x, y))
                cv2.rectangle(blurred_gray, (x - step, y - step), (x + step, y + step), 0, -1)  # Mask detected region

    # Draw bounding boxes for airglow and save the coordinates
    for region in bright_regions:
        top_left = (region[0] - args["radius"], region[1] - args["radius"])
        bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
        airglow_bboxes.append((top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue for airglow

    # -----------------------------
    #  STAR DETECTION (GREEN BOXES)
    # -----------------------------
    gray_no_blur = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)  # Original grayscale image
    laplacian = cv2.Laplacian(gray_no_blur, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)

    # Apply circular mask
    masked_sharpness = cv2.bitwise_and(sharpness_map, mask)

    # Adjusted threshold to detect slightly larger stars
star_threshold = avg_brightness * 3.5  # Lowered from 3.8 to 3.5 to detect slightly dimmer stars
detected_stars = []

# Detect bright sharp spots (stars)
for _ in range(7):  # Increase detection limit to catch slightly bigger stars
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(masked_sharpness)
    x, y = maxLoc

    # Ensure detection is inside the circular region
    if (x - center[0])**2 + (y - center[1])**2 < radius**2:
        if maxVal > star_threshold:
            # Make bounding box size proportional to brightness
            bbox_size = max(args["radius"] // 3, int(maxVal / 100))  # Dynamically scale bounding box

            top_left = (x - bbox_size, y - bbox_size)
            bottom_right = (x + bbox_size, y + bbox_size)

            # Check if the star bounding box overlaps with any airglow box
            overlap_found = False
            for airglow_bbox in airglow_bboxes:
                if is_overlap((top_left[0], top_left[1], bbox_size * 2, bbox_size * 2), airglow_bbox):
                    overlap_found = True
                    break

            # Only draw green bounding box if no overlap
            if not overlap_found:
                detected_stars.append((x, y))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green for stars

                # Mask out detected star to prevent repeated detection
                cv2.rectangle(masked_sharpness, top_left, bottom_right, 0, -1)

                top_left = (x - bbox_size, y - bbox_size)
                bottom_right = (x + bbox_size, y + bbox_size)

                # Check if the star bounding box overlaps with any airglow box
                overlap_found = False
                for airglow_bbox in airglow_bboxes:
                    if is_overlap((top_left[0], top_left[1], bbox_size*2, bbox_size*2), airglow_bbox):
                        overlap_found = True
                        break

                # Only draw green bounding box if no overlap
                if not overlap_found:
                    detected_stars.append((x, y))
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green for stars

                    # Mask out detected star to prevent repeated detection
                    cv2.rectangle(masked_sharpness, top_left, bottom_right, 0, -1)
    # -----------------------------
    #  SAVE PROCESSED IMAGE
    # -----------------------------
    output_image_path = os.path.join(bounded_image_folder, os.path.splitext(os.path.basename(image_path))[0] + "_processed.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {output_image_path}")

print("Processing complete!")
