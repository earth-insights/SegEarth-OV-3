import os
import cv2
import numpy as np

def process_dataset(root_dir):
    # Define paths for the new subdirectories
    image_dir = os.path.join(root_dir, 'image')
    label_dir = os.path.join(root_dir, 'label')

    # Create directories if they do not exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # List all files in the target directory
    files = os.listdir(root_dir)

    for filename in files:
        file_path = os.path.join(root_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue

        if filename.endswith('_sat.png'):
            # --- Handle Satellite Images ---
            # Remove '_sat' suffix and move to 'image' folder
            new_name = filename.replace('_sat', '')
            target_path = os.path.join(image_dir, new_name)
            
            # Read and save (or just move) the image
            img = cv2.imread(file_path)
            if img is not None:
                cv2.imwrite(target_path, img)
                print(f"Processed image: {filename} -> image/{new_name}")
                # Optional: remove the original file after processing
                # os.remove(file_path)

        elif filename.endswith('_mask.png'):
            # --- Handle Mask Images ---
            # Remove '_mask' suffix
            new_name = filename.replace('_mask', '')
            target_path = os.path.join(label_dir, new_name)

            # Read mask in grayscale
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # Binarization: > 128 set to 1, <= 128 set to 0
                # Using standard thresholding
                _, binary_mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
                
                # Save the processed mask
                cv2.imwrite(target_path, binary_mask)
                print(f"Processed mask: {filename} -> label/{new_name}")
                # Optional: remove the original file after processing
                # os.remove(file_path)

if __name__ == "__main__":
    # Change this to your actual directory path
    target_directory = 'data/GF_LowGradeRoadDataset/test' 
    process_dataset(target_directory)
    print("Dataset organization and mask processing complete.")