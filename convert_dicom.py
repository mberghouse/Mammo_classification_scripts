import os
import pydicom
import numpy as np
from PIL import Image

def apply_windowing(image_data, window_center, window_width):
    min_val = window_center - 0.5 - (window_width-1)/2
    max_val = window_center - 0.5 + (window_width-1)/2
    
    image_data = (image_data - min_val) / (max_val - min_val)
    image_data[image_data < 0] = 0
    image_data[image_data > 1] = 1
    image_data = (image_data * 255).astype('uint8')
    
    return image_data
    
def dicom_to_png(src_folder, dest_folder):
    # Ensure destination directory exists
    wc=1600
    ww=1500
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate over all the folders in source directory
    for dir_name in os.listdir(src_folder):
        full_dir_name = os.path.join(src_folder, dir_name)
        
        # Ensure it's actually a directory
        if os.path.isdir(full_dir_name):
            dest_dir = os.path.join(dest_folder, dir_name)
            
            # Ensure destination sub-directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Convert all DICOM files in this directory
            for filename in os.listdir(full_dir_name):
                if filename.endswith('.dicom'):
                    # Read the DICOM file
                    ds = pydicom.dcmread(os.path.join(full_dir_name, filename))
                    #if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    #    wc = int(ds.WindowCenter)
                    #    ww = int(ds.WindowWidth)
                    #    image_data = apply_windowing(ds.pixel_array, wc, ww)
                    #else:
                    #    image_data = ds.pixel_array
                    image_data = apply_windowing(ds.pixel_array, wc, ww)
                    # Convert to PNG
                    print (np.max(image_data), np.min(image_data))
                    print (image_data)
                    png_filename = os.path.join(dest_dir, filename.replace('.dicom', '.png'))
                    Image.fromarray(image_data).save(png_filename)

# Paths
src_folder = "vindr_extract/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
dest_folder = "600x1000"

# Convert all DICOM files
dicom_to_png(src_folder, dest_folder)

    

