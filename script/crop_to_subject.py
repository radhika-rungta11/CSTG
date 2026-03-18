# MIT License
# Copyright (c) 2023 OPPO
#
# Scans all cam_*/YYYYY.png alpha channels, finds the union bounding box of the
# foreground across every camera and every frame, adds padding, then crops all
# source images in-place to that uniform rectangle.
#
# After running, delete existing colmap_* folders and re-run pre_custom.py.
#
# Usage:
#   python script/crop_to_subject.py --datapath data/leopard --padding 0.05

import os
import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def find_alpha_bounds(path):
    """Return (rmin, rmax, cmin, cmax) of non-transparent pixels, or None."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] < 4:
        return None
    alpha = img[:, :, 3]
    if img.dtype == np.uint16:
        nonzero = alpha > 0
    else:
        nonzero = alpha > 0
    rows = np.any(nonzero, axis=1)
    cols = np.any(nonzero, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def crop_image(path, r1, r2, c1, c2):
    """Crop image at path to [r1:r2, c1:c2] and save in-place."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    cropped = img[r1:r2, c1:c2]
    cv2.imwrite(path, cropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True, type=str,
                        help="Path to dataset folder containing cam_*/ subdirs")
    parser.add_argument("--padding", default=0.05, type=float,
                        help="Fractional padding to add around the bounding box (default 0.05 = 5%%)")
    args = parser.parse_args()

    datapath = args.datapath.rstrip("/")
    camfolders = sorted(glob.glob(os.path.join(datapath, "cam_*/")))
    if not camfolders:
        print("No cam_*/ folders found in", datapath)
        raise SystemExit(1)

    # Collect all image paths
    all_paths = []
    for camfolder in camfolders:
        frames = sorted(glob.glob(os.path.join(camfolder, "*.png")))
        all_paths.extend(frames)

    print(f"Scanning {len(all_paths)} images across {len(camfolders)} cameras...")

    # Read one image to get dimensions
    sample = cv2.imread(all_paths[0], cv2.IMREAD_UNCHANGED)
    H, W = sample.shape[:2]
    print(f"Image size: {W}x{H}")

    # Find global union bounding box
    global_rmin, global_rmax, global_cmin, global_cmax = H, 0, W, 0

    for path in tqdm(all_paths, desc="Finding bounds"):
        bounds = find_alpha_bounds(path)
        if bounds is None:
            continue
        rmin, rmax, cmin, cmax = bounds
        global_rmin = min(global_rmin, rmin)
        global_rmax = max(global_rmax, rmax)
        global_cmin = min(global_cmin, cmin)
        global_cmax = max(global_cmax, cmax)

    print(f"Raw foreground bounds: rows [{global_rmin}, {global_rmax}], cols [{global_cmin}, {global_cmax}]")
    print(f"Raw crop size: {global_cmax - global_cmin + 1}x{global_rmax - global_rmin + 1}")

    # Add padding
    pad_r = int((global_rmax - global_rmin) * args.padding)
    pad_c = int((global_cmax - global_cmin) * args.padding)
    r1 = max(0, global_rmin - pad_r)
    r2 = min(H, global_rmax + pad_r + 1)
    c1 = max(0, global_cmin - pad_c)
    c2 = min(W, global_cmax + pad_c + 1)

    print(f"Padded crop box: rows [{r1}, {r2}), cols [{c1}, {c2})")
    print(f"Final crop size: {c2 - c1}x{r2 - r1}")

    # Crop all images in-place
    print("Cropping all images...")
    for path in tqdm(all_paths, desc="Cropping"):
        crop_image(path, r1, r2, c1, c2)

    print("\nDone.")
    print(f"All {len(all_paths)} images cropped to {c2-c1}x{r2-r1}.")
    print("\nNext steps:")
    print(f"  1. Delete existing colmap folders:  rm -rf {datapath}/colmap_*")
    print(f"  2. Re-run preprocessing:            python script/pre_custom.py --videopath {datapath} --startframe 0 --endframe 37")
