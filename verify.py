import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_annotations(image_path, label_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Read label file
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.split())
            x_min = (x - w / 2) * img.shape[1]
            y_min = (y - h / 2) * img.shape[0]
            width = w * img.shape[1]
            height = h * img.shape[0]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

    plt.show()

visualize_annotations("dataset/train/converted_images/PKR_DASC_0428_20160217_143735.846.png", "dataset/train/converted_images/PKR_DASC_0428_20160217_143735.846.txt")
