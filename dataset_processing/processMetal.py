import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Path to the folder containing the images and where to save .npy files, change as
data_folder = 'data'
new_data_npy = 'data_formatted/images_resized_npy'
new_data_jpg = 'data_formatted/images_resized_jpg'
new_data_npy_arr = 'data_formatted/images_resized_npy_arr'

# Create the new folder if it doesn't exist
if not os.path.exists(new_data_npy):
    os.makedirs(new_data_npy)
if not os.path.exists(new_data_jpg):
    os.makedirs(new_data_jpg)

# Create a list of all image files in the folder
image_files = [f for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

imgs = []  # List to store all processed images

# Loop through each image file with a progress bar
for image_file in tqdm(image_files, desc="Processing Images"):
    # Construct the full path to the image file
    img_path = os.path.join(data_folder, image_file)
    
    # Read the original image
    img = cv2.imread(img_path)
    
    # Check if the image was read correctly
    if img is None:
        print(f"Error reading image: {img_path}")
        continue

    # Resize the image while maintaining the aspect ratio
    if img.shape[0] > img.shape[1]:
        img_resized = cv2.resize(img, (int(img.shape[1] * (512 / img.shape[0])), 512))
    else:
        img_resized = cv2.resize(img, (512, int(img.shape[0] * (512 / img.shape[1]))))

    # Pad the image to make it 512x512 
    img_padded = cv2.copyMakeBorder(img_resized, 0, 512 - img_resized.shape[0], 0, 512 - img_resized.shape[1], cv2.BORDER_CONSTANT, value=0)
    # Save the image to get it export ready
    img_padded_path = os.path.join(new_data_jpg, image_file)
    # Uncomment if you want to save 
    # cv2.imwrite(img_padded_path, img_padded)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    
    # Uncomment if you need every image saved as npy
    # Save each processed image individually as a .npy file
    npy_filename = os.path.splitext(image_file)[0] + '.npy' # Create .npy filename
    npy_filepath = os.path.join(new_data_npy, npy_filename) # Full path for saving
    np.save(npy_filepath, img_gray)  # Save the processed image in npy format
    
    # Append the processed image to the list
    imgs.append(img_gray)

# Convert the list of images to a numpy array
imgs = np.array(imgs)
# UNCOMMENT as needed : Save the entire array of images as a single .npy files 
# full_imgs_filepath = os.path.join(new_data_npy_arr, 'imgs.npy')
# np.save(full_imgs_filepath, imgs)
print(f"All processed images saved individually as imgs.npy .")



# import cv2

# def resize_and_pad(img, target_size=512):
#     # Resize the image while maintaining the aspect ratio
#     if img.shape[0] > img.shape[1]:
#         img_resized = cv2.resize(img, (int(img.shape[1] * (target_size / img.shape[0])), target_size))
#     else:
#         img_resized = cv2.resize(img, (target_size, int(img.shape[0] * (target_size / img.shape[1]))))

#     # Calculate the padding needed to make it 512x512
#     h_pad = target_size - img_resized.shape[0]
#     w_pad = target_size - img_resized.shape[1]

#     # Calculate padding for top/bottom and left/right to be symmetrical
#     top_pad = h_pad // 2
#     bottom_pad = h_pad - top_pad
#     left_pad = w_pad // 2
#     right_pad = w_pad - left_pad

#     # Pad the image symmetrically
#     img_padded = cv2.copyMakeBorder(img_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

#     return img_padded