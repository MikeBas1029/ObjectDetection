import os

# Get the root directory of the project
project_root = os.path.dirname(os.path.abspath(__file__))

# Define dataset paths relative to the project root
train_images_path = os.path.join(project_root, "dataset/train/images")
val_images_path = os.path.join(project_root, "dataset/val/images")

print("Train images path:", train_images_path)
print("Validation images path:", val_images_path)
