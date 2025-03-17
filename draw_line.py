import os
import cv2
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN

def process_label_file(label_file):
    """ Reads label file and extracts bounding box centers and probabilities. """
    centers = []
    probabilities = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip malformed lines
            class_id, x, y, w, h, prob = map(float, parts)
            
            # Skip artifacts (class_id == 1), only keep airglow (class_id == 0)
            if class_id != 0:
                continue
            
            centers.append((x, y))
            probabilities.append(prob)
    return np.array(centers), np.array(probabilities)


def find_best_interest_direction(centers, probabilities, image_shape, prob_threshold=0.3):
    """ Finds the direction of most airglow regions using a weighted average. """
    mask = probabilities > prob_threshold
    filtered_centers = centers[mask]
    filtered_probabilities = probabilities[mask]
    
    if len(filtered_centers) == 0:
        # Fallback: No valid detections above threshold
        max_idx = np.argmax(probabilities)
        return centers[max_idx], True  # Return highest probability region
    
    # Normalize centers to be relative to image center
    image_center = np.array([0.5, 0.5])
    centered_points = filtered_centers - image_center
    
    # Cluster the filtered centers
    clustering = DBSCAN(eps=0.15, min_samples=2).fit(centered_points)
    labels = clustering.labels_
    
    unique_labels = np.unique(labels)
    cluster_directions = []
    cluster_weights = []
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        mask = labels == label
        cluster_centers = centered_points[mask]
        cluster_probs = filtered_probabilities[mask]
        
        # Calculate distance from image center for each point
        distances = np.linalg.norm(cluster_centers, axis=1)
        
        # Weight by both probability and distance from center
        weights = cluster_probs * distances
        weighted_centers = cluster_centers * weights[:, np.newaxis]
        avg_direction = np.sum(weighted_centers, axis=0) / np.sum(weights)
        
        cluster_directions.append(avg_direction)
        cluster_weights.append(np.sum(weights))
    
    if not cluster_directions:
        # Fallback: No clusters found
        max_idx = np.argmax(probabilities)
        return centers[max_idx], True  # Return highest probability region
    
    # Calculate final weighted average direction across clusters
    cluster_directions = np.array(cluster_directions)
    cluster_weights = np.array(cluster_weights)
    final_direction = np.sum(cluster_directions * cluster_weights[:, np.newaxis], axis=0) / np.sum(cluster_weights)
    
    # Normalize the direction vector and convert back to image coordinates
    final_direction_normalized = final_direction / np.linalg.norm(final_direction)
    final_direction_image_coords = final_direction_normalized + image_center
    
    # Find closest bounding box center to this direction for precise targeting
    distances_to_final_dir = np.linalg.norm(filtered_centers - final_direction_image_coords, axis=1)
    closest_idx = np.argmin(distances_to_final_dir)
    
    return filtered_centers[closest_idx], False  # Return closest bounding box center

def draw_on_image(image_file, target_point):
    """ Draws a line from image center to a target point (bounding box center). """
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    center_pixel_coords = (width // 2, height // 2)  # Image center
    
    # Convert normalized target point to pixel coordinates
    target_pixel_coords = (
        int(target_point[0] * width),
        int(target_point[1] * height),
    )
    
    cv2.line(image, center_pixel_coords, target_pixel_coords, (0, 0, 255), 2)  # Red line
    cv2.circle(image, target_pixel_coords, radius=5, color=(255, 0, 0), thickness=-1)  # Blue dot at target
    
    output_path = image_file.replace("dataset/train/bounded_images", "dataset/train/test_labels")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def main():
    label_files = glob("dataset/train/test_labels/*.txt")
    
    for label_file in label_files:
        print("Processing label file:", label_file)
        
        image_file = label_file.replace("test_labels", "bounded_images").replace(".txt", "_processed.jpg")
        
        if not os.path.exists(image_file):
            print(f"Image file not found: {image_file}")
            continue
        
        centers, probabilities = process_label_file(label_file)
        
        if len(centers) == 0:
            print(f"No valid detections in {label_file}")
            continue

        image_shape = cv2.imread(image_file).shape[:2]  # Get image shape (height, width)

        target_point, is_fallback = find_best_interest_direction(centers, probabilities, image_shape)

        draw_on_image(image_file, target_point)

        if is_fallback:
            print(f"Fallback used for {image_file}, pointing to region of highest probability.")
        else:
            print(f"Processed {image_file}, pointing to most likely airglow direction.")

if __name__ == "__main__":
    main()
