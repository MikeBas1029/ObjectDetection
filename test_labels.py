import os

train_images_dir = "dataset/train/converted_images"
train_labels_dir = "dataset/train/labels"
val_images_dir = "dataset/val/converted_images"
val_labels_dir = "dataset/val/labels"

print("Train Images:", len(os.listdir(train_images_dir)), "files")
print("Train Labels:", len(os.listdir(train_labels_dir)), "files")
print("Validation Images:", len(os.listdir(val_images_dir)), "files")
print("Validation Labels:", len(os.listdir(val_labels_dir)), "files")

# Check for mismatched files
for image_file in os.listdir(train_images_dir):
    label_file = os.path.join(train_labels_dir, image_file.replace(".png", ".txt"))
    if not os.path.exists(label_file):
        print("Missing label for:", image_file)
    else:
        print("Matched:", image_file)

for image_file in os.listdir(val_images_dir):
    label_file = os.path.join(val_labels_dir, image_file.replace(".png", ".txt"))
    if not os.path.exists(label_file):
        print("Missing label for:", image_file)
    else:
        print("Matched:", image_file)
