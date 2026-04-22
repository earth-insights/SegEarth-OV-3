import cv2
import os
import numpy as np

def process_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # List all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image (common formats)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            img_path = os.path.join(input_dir, filename)
            
            # Read the image in grayscale mode
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Skipping: {filename} (could not read)")
                continue

            # Apply thresholding:
            # If value >= 128, set to 1. Else, set to 0.
            # cv2.threshold returns (retval, thresholded_image)
            _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

            # Define the output path
            save_path = os.path.join(output_dir, filename)
            
            # Save the processed image
            # Note: The output image will look black in most viewers because pixels are 0 and 1
            cv2.imwrite(save_path, binary_img)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    # Define your paths here
    input_folder = 'data/GF7-building/Test/label'
    output_folder = 'data/GF7-building/Test/label_cvt'
    
    process_images(input_folder, output_folder)
    print("Done!")