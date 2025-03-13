import os
import cv2
import numpy as np

bounded_image_folder = "dataset/train/bounded_images"

def read_false_origin(file_path):
    """Read the false origin from a given file."""
    with open(file_path, 'r') as file:
        x, y = map(int, file.readline().strip().split(','))
    return (x, y)

def read_airglow_labels(file_path):
    """Read airglow regions and their probabilities from the label file."""
    airglow_regions = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            classification = int(data[0])
            if classification == 0:  # Only process airglow
                x, y, w, h, prob = map(float, data[1:])
                airglow_regions.append(((x, y), prob))
    return airglow_regions

def calculate_average_point(regions):
    """Calculate the average center of airglow regions."""
    x_avg = sum(x for (x, _), _ in regions) / len(regions)
    y_avg = sum(y for (_, y), _ in regions) / len(regions)
    return (x_avg, y_avg)

def are_regions_close(regions, threshold=0.05):
    """Determine if the detected airglow regions are close together."""
    if len(regions) < 2:
        return False
    x_coords = [x for (x, _), _ in regions]
    y_coords = [y for (_, y), _ in regions]
    return max(x_coords) - min(x_coords) < threshold and max(y_coords) - min(y_coords) < threshold

def draw_line_on_image(image_path, origin, target, output_path):
    """Draw a line from the false origin to the airglow target on the image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    h, w, _ = image.shape
    origin = (int(origin[0]), int(origin[1]))
    target = (int(target[0] * w), int(target[1] * h))
    
    cv2.line(image, origin, target, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def process_images(false_origins_path, test_labels_path):
    """Process each test label file and draw lines based on airglow detection."""
    for filename in os.listdir(test_labels_path):
        if filename.endswith('.txt'):
            false_origin_file = os.path.join(false_origins_path, filename)  # Match by filename
            test_label_file = os.path.join(test_labels_path, filename)

            '''change to for no bounding boxes in images:
            image_file = os.path.join(test_labels_path, filename.replace('.txt', '.png'))
            output_file = os.path.join(test_labels_path, filename.replace('.txt', '_output.png'))'''

            image_file = os.path.join(bounded_image_folder, filename.replace('.txt', '_processed.jpg'))
            output_file = os.path.join(test_labels_path, filename.replace('.txt', '.png'))
            
            if not os.path.exists(false_origin_file):
                print(f"Missing false origin file for {filename}, skipping.")
                continue

            false_origin = read_false_origin(false_origin_file)
            airglow_regions = read_airglow_labels(test_label_file)
            
            if not airglow_regions:
                continue
            
            if are_regions_close(airglow_regions):
                target = calculate_average_point(airglow_regions)
            else:
                target = max(airglow_regions, key=lambda x: x[1])[0]
            
            draw_line_on_image(image_file, false_origin, target, output_file)

process_images('dataset/train/false_origins', 'dataset/train/test_labels')