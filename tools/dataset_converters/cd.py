import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional

# Constants
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.tif')
SUB_DIRS = ['A', 'B', 'label']
DEFAULT_IMG_SIZE = 1024
DEFAULT_CROP_SIZE = 512


def process_single_image(
    filename: str,
    src_dir: str,
    dst_dir: str,
    img_size: int,
    crop_size: int,
    steps: int,
    is_mask: bool = False
) -> Optional[str]:
    """
    Process a single image: validate size, crop into tiles, and save.
    If is_mask is True, converts pixel values from 0-255 to 0-1.
    """
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        return None

    src_path = os.path.join(src_dir, filename)
    
    try:
        with Image.open(src_path) as img:
            # Ensure image is in L (Grayscale) mode for masks if necessary
            if is_mask and img.mode != 'L':
                img = img.convert('L')

            # Validate dimensions
            if img.size != (img_size, img_size):
                return f"Skipped {filename}: size is {img.size}"

            name, ext = os.path.splitext(filename)

            # Perform cropping
            for i in range(steps):       # Row
                for j in range(steps):   # Column
                    left = j * crop_size
                    upper = i * crop_size
                    right = left + crop_size
                    lower = upper + crop_size
                    
                    crop = img.crop((left, upper, right, lower))
                    
                    # Convert 0-255 to 0-1 for label masks
                    if is_mask:
                        # point() maps each pixel: if > 127 set to 1, else 0
                        # Alternatively, use: crop = crop.point(lambda p: 1 if p > 0 else 0)
                        crop = crop.point(lambda p: 1 if p > 127 else 0)
                    
                    # Generate filename and save
                    save_name = f"{name}_{i}_{j}{ext}"
                    crop.save(os.path.join(dst_dir, save_name))
                    
        return f"Success: {filename}"
    except Exception as e:
        return f"Error {filename}: {str(e)}"


def crop_images_multiprocess(
    input_root: str,
    output_root: str,
    max_workers: Optional[int] = None
) -> None:
    """
    Orchestrates the multi-process cropping task across sub-directories.
    """
    img_size = DEFAULT_IMG_SIZE
    crop_size = DEFAULT_CROP_SIZE
    steps = img_size // crop_size 

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for sub_dir in SUB_DIRS:
            src_dir = os.path.join(input_root, sub_dir)
            dst_dir = os.path.join(output_root, sub_dir)

            if not os.path.exists(src_dir):
                print(f"Skipping: {src_dir} does not exist")
                continue

            os.makedirs(dst_dir, exist_ok=True)
            files = os.listdir(src_dir)
            
            # Identify if the current directory contains masks
            is_mask = (sub_dir == "label")

            # Prepare the function with fixed arguments
            worker_func = partial(
                process_single_image, 
                src_dir=src_dir, 
                dst_dir=dst_dir, 
                img_size=img_size, 
                crop_size=crop_size, 
                steps=steps,
                is_mask=is_mask
            )

            print(f"Processing directory: {sub_dir} ({len(files)} files, is_mask={is_mask})")
            
            # Map tasks to the process pool
            results = list(executor.map(worker_func, files))
            
            # Summary statistics
            success_count = sum(1 for r in results if r and r.startswith("Success"))
            print(f"Finished {sub_dir}. Success: {success_count}/{len(files)}")


if __name__ == "__main__":
    # Configuration
    INPUT_PATH = "data/LEVIR-CD/test" # or WHU-CD
    OUTPUT_PATH = "data/LEVIR-CD/test_512" # or WHU-CD
    NUM_WORKERS = 32

    crop_images_multiprocess(INPUT_PATH, OUTPUT_PATH, max_workers=NUM_WORKERS)
    print("All tasks completed successfully.")