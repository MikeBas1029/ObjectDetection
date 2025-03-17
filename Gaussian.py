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
image_folder = "dataset/train/converted_images/tempfolder"

# Get image paths
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to check if two rectangles overlap
def is_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
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

    # Compute adaptive brightness threshold
    avg_brightness = np.mean(gray[mask == 255])
    min_brightness_threshold = np.percentile(gray[mask == 255], 30)

    bright_regions = []
    occupied_positions = set()
    step = args["radius"] * 2
    airglow_bboxes = []
    top_threshold = int(0.15 * height)

    if avg_brightness > 10:
        for _ in range(5):
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred_gray)
            x, y = maxLoc

            while (x, y) in occupied_positions:
                x += step
                if x >= image.shape[1]:
                    x = step
                    y += step
                if y >= image.shape[0]:
                    break

            if maxVal > avg_brightness * 1.2 and y > top_threshold:
                bright_regions.append((x, y))
                occupied_positions.add((x, y))

    for region in bright_regions:
        top_left = (region[0] - args["radius"], region[1] - args["radius"])
        bottom_right = (region[0] + args["radius"], region[1] + args["radius"])
        airglow_bboxes.append((top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

    # STAR DETECTION WITH ADAPTIVE SHARPNESS
    gray_no_blur = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_no_blur, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)
    masked_sharpness = cv2.bitwise_and(sharpness_map, mask)

    mean_sharpness = np.mean(masked_sharpness[mask == 255])
    std_sharpness = np.std(masked_sharpness[mask == 255])
    star_threshold = mean_sharpness + 2 * std_sharpness

    detected_stars = []
    min_star_distance = args["radius"] // 2

    for _ in range(7):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(masked_sharpness)
        x, y = maxLoc

        if (x - center[0])**2 + (y - center[1])**2 < radius**2:
            if maxVal > star_threshold and np.mean(gray[y-2:y+2, x-2:x+2]) > min_brightness_threshold:
                overlap_found = any(
                    np.linalg.norm(np.array(detected) - np.array((x, y))) < min_star_distance for detected in detected_stars
                )

                if not overlap_found:
                    detected_stars.append((x, y))
                    cv2.rectangle(masked_sharpness, (x-2, y-2), (x+2, y+2), 0, -1)

    output_txt_path = os.path.splitext(image_path)[0] + ".txt"
    with open(output_txt_path, "w") as f:
        for region in airglow_bboxes:
            x_center = (region[0] + region[2] / 2) / width
            y_center = (region[1] + region[3] / 2) / height
            w_norm = region[2] / width
            h_norm = region[3] / height
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
        for star in detected_stars:
            x_center = star[0] / width
            y_center = star[1] / height
            w_norm = (args["radius"] // 3 * 2) / width
            h_norm = (args["radius"] // 3 * 2) / height
            f.write(f"1 {x_center} {y_center} {w_norm} {h_norm}\n")

    output_image_path = os.path.join(image_folder, os.path.splitext(os.path.basename(image_path))[0] + ".png")
    cv2.imwrite(output_image_path, image)
    print(f"Saved processed image to {output_image_path}")

print("Processing complete!")
