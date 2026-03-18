# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Preprocessor for custom multi-camera datasets where:
#   - Frames are already extracted as cam_XXX/YYYYY.png (zero-padded 5-digit)
#   - No poses_bounds.npy or models.json (no known camera intrinsics/extrinsics)
#   - COLMAP runs full SfM (mapper) to estimate intrinsics + extrinsics from scratch
#
# Usage:
#   python script/pre_custom.py --videopath data/leopard --startframe 0 --endframe 37
#
# After running, set "duration": <endframe - startframe> in your training config.

import os
import glob
import tqdm
import sys
import argparse
import numpy as np
import cv2

sys.path.append(".")
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglecustom, triangulateperframe


def convertimage(imagepath):
    """Read and convert to 8-bit RGB. COLMAP requires 8-bit RGB input."""
    img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def preparecolmapframes(folder, offset=0):
    """Copy one frame per camera into colmap_<offset>/input/ for COLMAP."""
    folderlist = sorted(glob.glob(os.path.join(folder, "cam_*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "input")
    os.makedirs(savedir, exist_ok=True)
    for camfolder in folderlist:
        camname = os.path.basename(camfolder.rstrip("/"))
        imagepath = os.path.join(camfolder, str(offset).zfill(5) + ".png")
        if not os.path.exists(imagepath):
            print(f"warning: missing frame {imagepath}, skipping")
            continue
        imagesavepath = os.path.join(savedir, camname + ".png")
        if not os.path.exists(imagesavepath):
            img = convertimage(imagepath)
            if img is not None:
                cv2.imwrite(imagesavepath, img)


def prepareimagesonly(folder, offset):
    """For frames after frame 0: place RGB images in colmap_<offset>/images/.
    Camera poses come from colmap_0 (static rig), so no COLMAP needed.
    """
    folderlist = sorted(glob.glob(os.path.join(folder, "cam_*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "images")
    os.makedirs(savedir, exist_ok=True)

    ref_images_dir = os.path.join(folder, "colmap_0", "images")

    for camfolder in folderlist:
        camname = os.path.basename(camfolder.rstrip("/"))
        imagepath = os.path.join(camfolder, str(offset).zfill(5) + ".png")
        if not os.path.exists(imagepath):
            print(f"warning: missing frame {imagepath}, skipping")
            continue

        imagesavepath = os.path.join(savedir, camname + ".png")

        # Match size of colmap_0 undistorted image
        target_hw = None
        ref_path = os.path.join(ref_images_dir, camname + ".png")
        if os.path.exists(ref_path):
            ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            if ref is not None:
                target_hw = ref.shape[:2]  # (H, W)

        if not os.path.exists(imagesavepath):
            img = convertimage(imagepath)
            if img is not None:
                if target_hw is not None:
                    img = cv2.resize(img, (target_hw[1], target_hw[0]))
                cv2.imwrite(imagesavepath, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=37, type=int)

    args = parser.parse_args()
    videopath = args.videopath
    startframe = args.startframe
    endframe = args.endframe

    if startframe >= endframe:
        print("start frame must be smaller than end frame")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()

    if not videopath.endswith("/"):
        videopath = videopath + "/"

    duration = endframe - startframe
    print(f"Preparing {duration} frame(s) [{startframe}..{endframe-1}]")
    print(f"  → set \"duration\": {duration} in your training config")

    # Frame 0: full COLMAP SfM to recover camera poses
    if startframe == 0:
        print("preparing colmap input for frame 0")
        preparecolmapframes(videopath, 0)
        print("running full COLMAP SfM for frame 0 (feature extraction, matching, mapper, undistortion)")
        getcolmapsinglecustom(videopath, 0)

    # Frames 1+: place images, then triangulate per-frame point cloud
    remaining = range(max(startframe, 1), endframe)
    if remaining:
        print(f"preparing images for frames {remaining.start}–{remaining.stop - 1}")
        for offset in tqdm.tqdm(remaining):
            prepareimagesonly(videopath, offset)
        print(f"triangulating per-frame point clouds for frames {remaining.start}–{remaining.stop - 1}")
        for offset in tqdm.tqdm(remaining):
            triangulateperframe(videopath, offset)
