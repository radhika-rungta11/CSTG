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

import os
import glob
import tqdm
import shutil
import sys
import argparse
import numpy as np
import cv2

sys.path.append(".")
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglecustom


def convertimage(imagepath):
    """Read and convert to 8-bit RGB, return None if missing."""
    img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def preparecolmapframes(folder, offset=0):
    """Copy one frame per camera into colmap_<offset>/input/ for COLMAP.
    Converts 16-bit RGBA to 8-bit RGB as COLMAP requires 8-bit input.
    """
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
    """For frames after frame 0: just place converted images in colmap_<offset>/images/.
    Camera poses come from colmap_0, so no COLMAP needed for subsequent frames.
    Symlinks colmap_0/sparse/0/points3D.bin so the scene loader can build
    the combined temporal point cloud.
    """
    folderlist = sorted(glob.glob(os.path.join(folder, "cam_*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "images")
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
                # resize to match colmap_0 undistorted dimensions so rays align
                ref_path = os.path.join(folder, "colmap_0", "images", camname + ".png")
                if os.path.exists(ref_path):
                    ref = cv2.imread(ref_path)
                    if ref is not None:
                        img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
                cv2.imwrite(imagesavepath, img)

    # symlink sparse/0/points3D.bin from colmap_0 so the scene loader can read it
    sparse_dst = os.path.join(folder, "colmap_" + str(offset), "sparse", "0")
    os.makedirs(sparse_dst, exist_ok=True)
    src = os.path.abspath(os.path.join(folder, "colmap_0", "sparse", "0", "points3D.bin"))
    dst = os.path.join(sparse_dst, "points3D.bin")
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)


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

    # Frame 0: full COLMAP SfM to recover camera poses
    if startframe == 0:
        print("preparing colmap input for frame 0")
        preparecolmapframes(videopath, 0)
        print("running full COLMAP SfM for frame 0 (feature extraction, matching, mapper, undistortion)")
        getcolmapsinglecustom(videopath, 0)

    # Frames 1+: cameras are static, just place images — no COLMAP needed
    remaining = range(max(startframe, 1), endframe)
    if remaining:
        print(f"preparing images for frames {remaining.start}-{remaining.stop - 1} (no COLMAP needed, using poses from frame 0)")
        for offset in tqdm.tqdm(remaining):
            prepareimagesonly(videopath, offset)
