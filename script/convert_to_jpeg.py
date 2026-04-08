#!/usr/bin/env python3
"""Convert all PNGs in cam_*/ folders to JPEG and remove originals."""

import os
import sys
import cv2
import glob
import argparse
from tqdm import tqdm


def convert_folder(folder, quality=95):
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not pngs:
        return 0
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    for png_path in pngs:
        img = cv2.imread(png_path)
        if img is None:
            print(f"  warning: failed to read {png_path}")
            continue
        jpg_path = os.path.splitext(png_path)[0] + ".jpg"
        cv2.imwrite(jpg_path, img, params)
        os.remove(png_path)
    return len(pngs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images to JPEG")
    parser.add_argument("datadir", help="Dataset directory (e.g. data/leopard_run)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    args = parser.parse_args()

    cam_folders = sorted(glob.glob(os.path.join(args.datadir, "cam_*/")))
    if not cam_folders:
        cam_folders = sorted(glob.glob(os.path.join(args.datadir, "cam[0-9]*/")))
    print(f"Found {len(cam_folders)} camera folders in {args.datadir}")

    total = 0
    for cam in tqdm(cam_folders, desc="Converting"):
        total += convert_folder(cam, args.quality)
    print(f"Converted {total} images to JPEG (quality={args.quality})")
