import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
train_fits_path = os.path.join(project_root, "dataset/train/images")
train_png_path = os.path.join(project_root, "dataset/train/converted_images")
val_fits_path = os.path.join(project_root, "dataset/val/images")
val_png_path = os.path.join(project_root, "dataset/val/converted_images")


def data_loader(input_dir, output_dir):

    # Iterate over all FITS files in the input directory
    for fits_file in os.listdir(input_dir):
        if fits_file.endswith(".FITS"):
            output_file = fits_file.replace(".FITS", ".png")  # Corresponding PNG file name
            output_path = os.path.join(output_dir, output_file)
            
            # Check if the PNG file already exists
            if os.path.exists(output_path):
                print(f"Skipping {fits_file}, {output_file} already exists.")
                continue
            
            fits_path = os.path.join(input_dir, fits_file)
            try:
                # Open FITS file and load data
                with fits.open(fits_path) as hdul:
                    data = hdul[0].data.astype(np.float32)  # Convert to float32 for processing

                # Normalize the data for visualization
                data = np.nan_to_num(data)  # Replace NaNs with zeros
                data -= data.min()  # Shift to zero
                if data.max() > 0:  # Avoid division by zero
                    data /= data.max()  # Normalize to [0, 1]
                data *= 255  # Scale to [0, 255]
                
                # Save as PNG
                plt.imsave(output_path, data, cmap="gray")
            except Exception as e:
                print(f"Error converting {fits_file}: {e}")

if __name__ == "__main__":
    # Convert FITS files to PNG for both training and validation datasets
    data_loader(train_fits_path, train_png_path)
    #data_loader(val_fits_path, val_png_path)
