import os

# Path to your images
image_dir = "dataset/train/converted_images"

# List all .png images in the directory
for image_name in os.listdir(image_dir):
    if image_name.endswith(".png"):
        # Generate the corresponding .txt filename
        txt_file = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(image_dir, txt_file)

        # Check if the .txt file already exists
        if not os.path.exists(txt_path):
            # Create an empty .txt file (you can add labels manually later)
            with open(txt_path, "w") as f:
                pass
        else:
            print(f"Skipping {image_name}, .txt file already exists.")