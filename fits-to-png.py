#%% 
import numpy as np 
from astropy.io import fits

# Code to convert fits images into pngs 
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
import os

#%%
def fits_to_png(fits_file, output_png):
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data

    # Normalize the image data using ZScale
    norm = ImageNormalize(image_data, interval=ZScaleInterval())

    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data, cmap='gray', norm=norm)
    plt.axis('off')  # Hide axes

    # Save the figure as a PNG file
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
    plt.close()
#%%
if __name__ == "__main__":
    # input directory 
    input_dir = "./data/251214/fits"
    output_dir = "./data/251214/pngs"
 
    os.makedirs(output_dir, exist_ok=True)
    fits_files = [f for f in os.listdir(input_dir) if f.endswith('.fits')]
    for fits_file in fits_files:
        fits_path = os.path.join(input_dir, fits_file)
        png_filename = fits_file.replace('.fits', '.png')
        output_png_path = os.path.join(output_dir, png_filename)
        # accept an OSerror and and pass the fil name 
        try:
            fits_to_png(fits_path, output_png_path)
            print(f"Converted {fits_file} to {png_filename}")
        except OSError as e:
            print(f"Error converting {fits_file}: {e}")
            # Write filename to a log file
            with open("conversion_errors.log", "a") as log_file:
                log_file.write(f"{fits_file}\n")
#%%
    # Take log file and sort to file names in order 
    with open("conversion_errors.log", "r") as log_file:
        error_files = log_file.readlines()
    error_files = [f.strip() for f in error_files]
    error_files.sort()
    with open("conversion_errors_sorted.log", "w") as sorted_log_file:
        for f in error_files:
            sorted_log_file.write(f"{f}\n")

# %%
