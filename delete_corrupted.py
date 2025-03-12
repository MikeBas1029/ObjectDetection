import os

folder_path = "dataset/train/converted_images" 

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        txt_path = os.path.join(folder_path, filename)
        with open(txt_path, 'r') as file:
            if file.read().strip() == "":  # Check if the file is empty
                imageName = filename[:-4] + ".png"
                os.remove(txt_path)
                os.remove(os.path.join(folder_path, imageName))
                print(f"Deleted empty file: {filename}")
                print(f"Deleted corresponding image: {imageName}")
