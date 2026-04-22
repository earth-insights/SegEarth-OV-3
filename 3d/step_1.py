import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mmseg.structures import SegDataSample
from segearthov3_segmentor import SegEarthOV3Segmentation

"""
[Split Step 1] 2D Inference Script
Function: Load images, run large model inference, and save the results as 8-bit single-channel PNG mask images.
"""

# Optimized streamlined prompt (significantly reduces attention computation)
STPLS3D_CLASSES = [
    'ground,road,grass,dirt,water',   # 0
    'tree,vegetation',                # 1
    'car,truck',                      # 2
    'light pole',                     # 3
    'fence',                          # 4
    'building,clutter'                # 5
]

def init_ovss_model(classname_txt_path='./configs/stpls3d_classes.txt'):
    os.makedirs(os.path.dirname(classname_txt_path), exist_ok=True)
    with open(classname_txt_path, 'w') as writers:
        for i, cls_str in enumerate(STPLS3D_CLASSES):
            writers.write(cls_str if i == len(STPLS3D_CLASSES)-1 else cls_str + '\n')
            
    print(f"[*] Initializing SegEarthOV3 model...")
    model = SegEarthOV3Segmentation(
        type='SegEarthOV3Segmentation', model_type='SAM3',
        classname_path=classname_txt_path, prob_thd=0.1, confidence_threshold=0.1, bg_idx=5
    )
    return model

def main():
    IMG_DIR = "dataset/3D/WMSC"
    OUT_MASKS_DIR = "./work_dirs/WMSC_2d_masks"
    
    os.makedirs(OUT_MASKS_DIR, exist_ok=True)
    model = init_ovss_model()
    
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))]
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMG_DIR, img_name)
        mask_filename = os.path.splitext(img_name)[0] + ".png"
        out_mask_path = os.path.join(OUT_MASKS_DIR, mask_filename)
        
        # If already exists, skip to support breakpoint resume
        if os.path.exists(out_mask_path):
            print(f"  [{idx+1}/{len(image_files)}] {img_name} mask already exists, skipping.")
            continue
            
        print(f"  [{idx+1}/{len(image_files)}] Inference: {img_name}")
        
        # 1. Resize image
        img = Image.open(img_path)
        new_w, new_h = int(img.size[0]), int(img.size[1])
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # 2. Construct tensor and metadata
        img_tensor = transforms.Compose([transforms.ToTensor()])(img_resized).unsqueeze(0).to('cuda')
        data_sample = SegDataSample()
        data_sample.set_metainfo({'img_path': img_path, 'ori_shape': (new_h, new_w)})
        
        # 3. Model prediction
        with torch.no_grad():
            seg_pred = model.predict(img_tensor, data_samples=[data_sample])
            
        # 4. Get 2D Label Map (uint8 format, range 0~5)
        label_map_2d = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0).astype(np.uint8)
        
        # 5. Save as grayscale image (PNG is lossless compression, suitable for masks)
        Image.fromarray(label_map_2d).save(out_mask_path)
        
    print("\n✅ All 2D inferences completed, masks saved to:", OUT_MASKS_DIR)

if __name__ == "__main__":
    main()