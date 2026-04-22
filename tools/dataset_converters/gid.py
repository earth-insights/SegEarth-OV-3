import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# 3. Specify to process only test set images
TEST_IMAGES = {
    "GF2_PMS1__L1A0001064454-MSS1.tif",
    "GF2_PMS1__L1A0001118839-MSS1.tif",
    "GF2_PMS1__L1A0001344822-MSS1.tif",
    "GF2_PMS1__L1A0001348919-MSS1.tif",
    "GF2_PMS1__L1A0001366278-MSS1.tif",
    "GF2_PMS1__L1A0001366284-MSS1.tif",
    "GF2_PMS1__L1A0001395956-MSS1.tif",
    "GF2_PMS1__L1A0001432972-MSS1.tif",
    "GF2_PMS1__L1A0001670888-MSS1.tif",
    "GF2_PMS1__L1A0001680857-MSS1.tif",
    "GF2_PMS1__L1A0001680858-MSS1.tif",
    "GF2_PMS1__L1A0001757429-MSS1.tif",
    "GF2_PMS1__L1A0001765574-MSS1.tif",
    "GF2_PMS2__L1A0000607677-MSS2.tif",
    "GF2_PMS2__L1A0000607681-MSS2.tif",
    "GF2_PMS2__L1A0000718813-MSS2.tif",
    "GF2_PMS2__L1A0001038935-MSS2.tif",
    "GF2_PMS2__L1A0001038936-MSS2.tif",
    "GF2_PMS2__L1A0001119060-MSS2.tif",
    "GF2_PMS2__L1A0001367840-MSS2.tif",
    "GF2_PMS2__L1A0001378491-MSS2.tif",
    "GF2_PMS2__L1A0001378501-MSS2.tif",
    "GF2_PMS2__L1A0001396036-MSS2.tif",
    "GF2_PMS2__L1A0001396037-MSS2.tif",
    "GF2_PMS2__L1A0001416129-MSS2.tif",
    "GF2_PMS2__L1A0001471436-MSS2.tif",
    "GF2_PMS2__L1A0001517494-MSS2.tif",
    "GF2_PMS2__L1A0001591676-MSS2.tif",
    "GF2_PMS2__L1A0001787564-MSS2.tif",
    "GF2_PMS2__L1A0001821754-MSS2.tif"
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_nirrgb_tif_extract_rgb(tif_path: Path) -> np.ndarray:
    """
    1. Process GF-2 4-channel image and extract RGB.
    GF-2 MSS sensor bands are usually: 1-Blue(B), 2-Green(G), 3-Red(R), 4-Near-Infrared(NIR).
    To obtain standard RGB images, we need to extract indices 2(R), 1(G), 0(B).
    """
    img = Image.open(str(tif_path))
    arr = np.array(img)  # HxWx4
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expect multi-band image at {tif_path}, got shape {arr.shape}")
    
    # Extract R, G, B channels
    rgb = arr[:, :, [1,2,3]] 
    return rgb


def read_index_mask(mask_path: Path) -> np.ndarray:
    """
    4. Read mask without modifying index order.
    """
    m = Image.open(str(mask_path))
    arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def sliding_window_coords(h, w, patch, stride):
    """
    Generate sliding window coordinates, ensuring the whole image is covered.
    """
    ys = list(range(0, max(h - patch + 1, 1), stride))
    xs = list(range(0, max(w - patch + 1, 1), stride))

    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]

    if ys[-1] != h - patch and h >= patch:
        ys.append(h - patch)
    if xs[-1] != w - patch and w >= patch:
        xs.append(w - patch)

    if h < patch:
        ys = [0]
    if w < patch:
        xs = [0]

    for y in ys:
        for x in xs:
            y1, x1 = y, x
            y2, x2 = min(y1 + patch, h), min(x1 + patch, w)
            yield y1, y2, x1, x2


def pad_to_patch(img: np.ndarray, patch: int, pad_val=0) -> np.ndarray:
    """
    Padding function
    """
    if img.ndim == 3:
        h, w, c = img.shape
        out = np.full((patch, patch, c), pad_val, dtype=img.dtype)
        out[:h, :w, :] = img
        return out
    else:
        h, w = img.shape
        out = np.full((patch, patch), pad_val, dtype=img.dtype)
        out[:h, :w] = img
        return out


def main():
    parser = argparse.ArgumentParser("Preprocess GF-2 Test Set: extract RGB & crop image into patches")

    parser.add_argument("--data-root", type=str, required=True,
                        help="GID root, e.g. /data/GID")
    parser.add_argument("--img-dir", type=str, default="Image__8bit_NirRGB",
                        help="subdir for NirRGB images")
    parser.add_argument("--mask-dir", type=str, default="Annotation__index",
                        help="subdir for index masks")
    parser.add_argument("--img-suffix", type=str, default=".tif",
                        help="image suffix")
    parser.add_argument("--mask-suffix", type=str, default="_5label.png",
                        help="mask suffix to match image stem")

    parser.add_argument("--out-root", type=str, required=True,
                        help="output root dir to save patches")
    parser.add_argument("--patch", type=int, default=1024,
                        help="patch size")
    parser.add_argument("--stride", type=int, default=1024,
                        help="stride for test set prediction (often same as patch size)")
    parser.add_argument("--pad", action="store_true", default=True,
                        help="pad border patches to patch size (recommended)")
    parser.add_argument("--ignore-index", type=int, default=255,
                        help="ignore index in GT padding")
    parser.add_argument("--save-rgb-full", action="store_true",
                        help="also save extracted full RGB image (not cropped) into out_root/full_rgb")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    img_dir = data_root / args.img_dir
    mask_dir = data_root / args.mask_dir

    out_root = Path(args.out_root)
    out_img = out_root / "images"
    ensure_dir(out_img)
    
    # For test set, if mask_dir is provided, also create a labels directory for use
    out_msk = out_root / "labels"
    ensure_dir(out_msk)

    if args.save_rgb_full:
        out_full = out_root / "rgb"
        ensure_dir(out_full)

    # Collect images (keep only those in the test set list)
    all_img_paths = sorted(img_dir.glob(f"*{args.img_suffix}"))
    img_paths = [p for p in all_img_paths if p.name in TEST_IMAGES]

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No test images found in {img_dir}. Check file names and suffix.")

    print(f"Found {len(img_paths)} test images to process.")

    total = 0

    for img_path in tqdm(img_paths, desc="Processing"):
        stem = img_path.stem 
        mask_path = mask_dir / f"{stem}{args.mask_suffix}"
        
        # Allow test set without GT labels. If it exists, process it; otherwise, skip the mask and only process the image.
        has_mask = mask_path.exists()

        # 1) Read and extract RGB
        rgb_u8 = read_nirrgb_tif_extract_rgb(img_path)
        h, w = rgb_u8.shape[:2]

        # 2) If mask exists, read it (without changing its internal index order)
        if has_mask:
            msk = read_index_mask(mask_path)
            if msk.shape[0] != h or msk.shape[1] != w:
                raise ValueError(f"Shape mismatch: {img_path} rgb={rgb_u8.shape[:2]} mask={msk.shape}")

        # Save an additional full-sized image
        if args.save_rgb_full:
            Image.fromarray(rgb_u8).save(str(out_full / f"{stem}.png"))

        # 3) Sliding window cropping
        for (y1, y2, x1, x2) in sliding_window_coords(h, w, args.patch, args.stride):
            img_patch = rgb_u8[y1:y2, x1:x2, :]
            if has_mask:
                msk_patch = msk[y1:y2, x1:x2]

            if args.pad and (img_patch.shape[0] != args.patch or img_patch.shape[1] != args.patch):
                img_patch = pad_to_patch(img_patch, args.patch, pad_val=0)
                if has_mask:
                    msk_patch = pad_to_patch(msk_patch, args.patch, pad_val=args.ignore_index)

            total += 1

            # Generate filename and save
            patch_name = f"{stem}_{x1}_{y1}_{args.patch}_{args.patch}.png"
            Image.fromarray(img_patch).save(str(out_img / patch_name))
            
            if has_mask:
                Image.fromarray(msk_patch).save(str(out_msk / patch_name))

    print(f"Done. total test windows = {total}")
    print(f"Patches saved to:\n  images: {out_img}")
    if out_msk.exists() and any(out_msk.iterdir()):
        print(f"  masks : {out_msk}")
    if args.save_rgb_full:
        print(f"Full RGB saved to: {out_full}")


if __name__ == "__main__":
    main()