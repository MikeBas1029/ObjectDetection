import os
import random
import shutil

# Paths
train_path = "dataset/train/converted_images"  # Current folder with all images
val_path = "dataset/val/converted_images"  # Folder where 30% will be moved

# Create validation directory if not exists
os.makedirs(val_path, exist_ok=True)

# Get all image filenames
all_images = [f for f in os.listdir(train_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle for randomness
random.seed(42)  # Ensure reproducibility
random.shuffle(all_images)

# Calculate 30% split
num_val = int(0.3 * len(all_images))
val_images = all_images[:num_val]

# Move selected images to validation folder
for img in val_images:
    shutil.move(os.path.join(train_path, img), os.path.join(val_path, img))

print(f"Moved {num_val} images to validation set.")