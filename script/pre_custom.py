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
import re
import shutil
import sqlite3

sys.path.append(".")
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglecustom, triangulateperframe
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec


def convertimage(imagepath, white_background=True):
    """Read and convert to 8-bit RGB. COLMAP requires 8-bit RGB input.
    For RGBA images, composites over white background so transparent pixels
    become white rather than black — avoids dark blob artifacts during training.
    """
    img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        if white_background:
            bg = np.ones_like(rgb) * 255.0
        else:
            bg = np.zeros_like(rgb)
        img = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return img


def extractforegroundmask(imagepath, target_hw=None):
    """Return 8-bit mask: 255 where foreground (alpha > 0), 0 where background.
    Resizes to target_hw=(H, W) if given. Returns None if no alpha channel.
    """
    src = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    if src is None or src.ndim < 3 or src.shape[2] < 4:
        return None
    alpha = src[:, :, 3]
    if src.dtype == np.uint16:
        alpha = (alpha / 256).astype(np.uint8)
    mask = (alpha > 0).astype(np.uint8) * 255
    if target_hw is not None and mask.shape[:2] != tuple(target_hw):
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def savemasks(folder, offset):
    """Save alpha masks for training to colmap_<offset>/masks/.
    Called after COLMAP undistortion (frame 0) or after prepareimagesonly (frames 1+).
    Mask names match colmap_<offset>/images/ and are resized to the same resolution.
    """
    imagesdir = os.path.join(folder, "colmap_" + str(offset), "images")
    maskdir = os.path.join(folder, "colmap_" + str(offset), "masks")
    if not os.path.isdir(imagesdir):
        return
    os.makedirs(maskdir, exist_ok=True)

    for imgpath in sorted(glob.glob(os.path.join(imagesdir, "cam_*.png"))):
        camname = os.path.basename(imgpath)          # cam_000.png
        cambase = os.path.splitext(camname)[0]       # cam_000
        masksavepath = os.path.join(maskdir, camname)
        if os.path.exists(masksavepath):
            continue
        srcpath = os.path.join(folder, cambase, str(offset).zfill(5) + ".png")
        if not os.path.exists(srcpath):
            continue
        ref = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        target_hw = ref.shape[:2] if ref is not None else None
        mask = extractforegroundmask(srcpath, target_hw=target_hw)
        if mask is not None:
            cv2.imwrite(masksavepath, mask)


def extractvideos(folder, startframe, endframe):
    """Extract frames from cam*.mp4 files into cam_XXX/YYYYY.png structure.
    cam00.mp4 -> cam_000/, cam01.mp4 -> cam_001/, etc.
    Skips files that already exist.
    """
    mp4s = sorted(glob.glob(os.path.join(folder, "cam*.mp4")))
    if not mp4s:
        print("no cam*.mp4 files found, skipping extraction")
        return
    print(f"extracting frames {startframe}..{endframe-1} from {len(mp4s)} video(s)")
    for mp4path in tqdm.tqdm(mp4s):
        basename = os.path.splitext(os.path.basename(mp4path))[0]  # e.g. "cam00"
        digits = re.sub(r"[^0-9]", "", basename)
        camname = "cam_" + digits.zfill(3)                         # e.g. "cam_000"
        outdir = os.path.join(folder, camname)
        os.makedirs(outdir, exist_ok=True)
        cap = cv2.VideoCapture(mp4path)
        fidx = 0
        extracted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if startframe <= fidx < endframe:
                outpath = os.path.join(outdir, str(fidx).zfill(5) + ".png")
                if not os.path.exists(outpath):
                    cv2.imwrite(outpath, frame)
                extracted += 1
            elif fidx >= endframe:
                break
            fidx += 1
        cap.release()


def preparecolmapframes(folder, offset=0):
    """Copy one frame per camera into colmap_<offset>/input/ for COLMAP.
    Also saves foreground masks to colmap_<offset>/input_masks/ so COLMAP
    ignores background pixels during feature extraction.
    """
    folderlist = sorted(glob.glob(os.path.join(folder, "cam_*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "input")
    maskdir = os.path.join(folder, "colmap_" + str(offset), "input_masks")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(maskdir, exist_ok=True)
    for camfolder in folderlist:
        camname = os.path.basename(camfolder.rstrip("/"))
        imagepath = os.path.join(camfolder, str(offset).zfill(5) + ".png")
        if not os.path.exists(imagepath):
            print(f"warning: missing frame {imagepath}, skipping")
            continue
        imagesavepath = os.path.join(savedir, camname + ".png")
        masksavepath = os.path.join(maskdir, camname + ".png.png")
        if not os.path.exists(imagesavepath):
            img = convertimage(imagepath)
            if img is not None:
                cv2.imwrite(imagesavepath, img)
        if not os.path.exists(masksavepath):
            mask = extractforegroundmask(imagepath)
            if mask is not None:
                cv2.imwrite(masksavepath, mask)


def prepareimagesonly(folder, offset):
    """For frames after frame 0: place RGB images in colmap_<offset>/images/
    and foreground masks in colmap_<offset>/input_masks/ for triangulation.
    Camera poses come from colmap_0 (static rig), so no COLMAP mapper needed.
    """
    folderlist = sorted(glob.glob(os.path.join(folder, "cam_*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "images")
    maskdir = os.path.join(folder, "colmap_" + str(offset), "input_masks")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(maskdir, exist_ok=True)

    ref_images_dir = os.path.join(folder, "colmap_0", "images")

    for camfolder in folderlist:
        camname = os.path.basename(camfolder.rstrip("/"))
        imagepath = os.path.join(camfolder, str(offset).zfill(5) + ".png")
        if not os.path.exists(imagepath):
            print(f"warning: missing frame {imagepath}, skipping")
            continue

        imagesavepath = os.path.join(savedir, camname + ".png")
        masksavepath = os.path.join(maskdir, camname + ".png.png")

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

        if not os.path.exists(masksavepath):
            mask = extractforegroundmask(imagepath, target_hw=target_hw)
            if mask is not None:
                cv2.imwrite(masksavepath, mask)

        # Training mask (same resolution as images/)
        trainmaskdir = os.path.join(folder, "colmap_" + str(offset), "masks")
        os.makedirs(trainmaskdir, exist_ok=True)
        trainmasksavepath = os.path.join(trainmaskdir, camname + ".png")
        if not os.path.exists(trainmasksavepath):
            trainmask = extractforegroundmask(imagepath, target_hw=target_hw)
            if trainmask is not None:
                cv2.imwrite(trainmasksavepath, trainmask)


def regeneratetrainimages(folder, startframe, endframe):
    """Re-composite training images over white background without re-running COLMAP.
    Use after changing convertimage() when COLMAP poses are already computed.
    Only rewrites colmap_<N>/images/ files; does not touch sparse/ or db.
    """
    ref_images_dir = os.path.join(folder, "colmap_0", "images")
    for offset in tqdm.tqdm(range(startframe, endframe)):
        imagesdir = os.path.join(folder, "colmap_" + str(offset), "images")
        if not os.path.isdir(imagesdir):
            continue
        for imgpath in sorted(glob.glob(os.path.join(imagesdir, "cam_*.png"))):
            camname = os.path.basename(imgpath)
            cambase = os.path.splitext(camname)[0]
            srcpath = os.path.join(folder, cambase, str(offset).zfill(5) + ".png")
            if not os.path.exists(srcpath):
                continue
            # Get target size from current undistorted image
            ref = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            target_hw = ref.shape[:2] if ref is not None else None
            img = convertimage(srcpath)
            if img is not None:
                if target_hw is not None and img.shape[:2] != target_hw:
                    img = cv2.resize(img, (target_hw[1], target_hw[0]))
                cv2.imwrite(imgpath, img)
        # Regenerate training masks too
        savemasks(folder, offset)


def colmapfromposesbounds(folder, offset):
    """Run COLMAP for one frame using ground-truth poses/intrinsics from poses_bounds.npy.

    COLMAP 3.x compatible: feature_extractor runs first on a fresh DB (so it creates
    the new rigs/frames schema), then the DB is patched with known calibration from
    poses_bounds.npy before triangulation. Avoids the duplicate-camera crash caused
    by pre-populating the DB before feature extraction.
    """
    posespath = os.path.join(folder, "poses_bounds.npy")
    poses_bounds = np.load(posespath)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    llffposes = poses.copy().transpose(1, 2, 0)
    w2c_matriclist = posetow2c_matrcs(llffposes)

    projectfolder = os.path.join(folder, "colmap_" + str(offset))
    dbfile = os.path.join(projectfolder, "input.db")
    inputimagefolder = os.path.join(projectfolder, "input")
    distortedmodel = os.path.join(projectfolder, "distorted/sparse")
    manualfolder = os.path.join(projectfolder, "manual")

    os.makedirs(distortedmodel, exist_ok=True)
    os.makedirs(manualfolder, exist_ok=True)

    # Delete any existing DB so COLMAP creates a fresh 3.x-compatible schema
    if os.path.exists(dbfile):
        os.remove(dbfile)

    # Step 1: feature extraction (creates DB with correct rig/frame tables)
    exit_code = os.system(
        "colmap feature_extractor --database_path " + dbfile +
        " --image_path " + inputimagefolder)
    if exit_code != 0:
        exit(exit_code)

    # Step 2: patch DB with ground-truth poses and intrinsics from poses_bounds.npy.
    # Match by position: images sorted alphabetically map to poses_bounds in cam-source order.
    # This is robust regardless of naming convention (cam00 vs cam_000 etc.).
    db = sqlite3.connect(dbfile)
    rows = db.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY name").fetchall()

    if len(rows) != len(poses):
        print(f"WARNING: DB has {len(rows)} images but poses_bounds has {len(poses)} cameras — "
              "results may be incorrect")

    imagetxtlist = []
    cameratxtlist = []

    for idx, (image_id, imgname, camera_id) in enumerate(rows):
        if idx >= len(poses):
            print(f"WARNING: more images than poses, skipping {imgname}")
            continue
        m = w2c_matriclist[idx]
        colmapR = m[:3, :3]
        T = m[:3, 3]
        H, W, focal = poses[idx, :, -1]
        colmapQ = rotmat2qvec(colmapR)

        params = np.array([focal, focal, W / 2.0, H / 2.0], dtype=np.float64)
        db.execute(
            "UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 "
            "WHERE camera_id=?",
            (int(W), int(H), params.tobytes(), camera_id))

        imagetxtlist.append(
            str(image_id) + " " +
            " ".join(str(colmapQ[j]) for j in range(4)) + " " +
            " ".join(str(T[j]) for j in range(3)) + " " +
            str(camera_id) + " " + imgname + "\n\n")
        cameratxtlist.append(
            str(camera_id) + " PINHOLE " + str(int(W)) + " " + str(int(H)) + " " +
            str(focal) + " " + str(focal) + " " +
            str(W / 2.0) + " " + str(H / 2.0) + "\n")

    db.commit()
    db.close()

    with open(os.path.join(manualfolder, "images.txt"), "w") as f:
        f.writelines(imagetxtlist)
    with open(os.path.join(manualfolder, "cameras.txt"), "w") as f:
        f.writelines(cameratxtlist)
    open(os.path.join(manualfolder, "points3D.txt"), "w").close()

    # Step 3: feature matching
    exit_code = os.system(
        "colmap exhaustive_matcher --database_path " + dbfile)
    if exit_code != 0:
        exit(exit_code)

    # Step 4: triangulate with known poses
    exit_code = os.system(
        "colmap point_triangulator --database_path " + dbfile +
        " --image_path " + inputimagefolder +
        " --output_path " + distortedmodel +
        " --input_path " + manualfolder +
        " --Mapper.ba_global_function_tolerance=0.000001")
    if exit_code != 0:
        exit(exit_code)

    # Step 5: undistort images
    exit_code = os.system(
        "colmap image_undistorter --image_path " + inputimagefolder +
        " --input_path " + distortedmodel +
        " --output_path " + projectfolder +
        " --output_type COLMAP")
    if exit_code != 0:
        exit(exit_code)

    os.system("rm -r " + inputimagefolder)

    # Move sparse files into sparse/0/
    sparsetop = os.path.join(projectfolder, "sparse")
    os.makedirs(os.path.join(sparsetop, "0"), exist_ok=True)
    for fname in os.listdir(sparsetop):
        if fname == "0":
            continue
        shutil.move(os.path.join(sparsetop, fname),
                    os.path.join(sparsetop, "0", fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=37, type=int)
    parser.add_argument("--regen-images", action="store_true",
                        help="Re-composite training images over white bg without re-running COLMAP")
    parser.add_argument("--masks-only", action="store_true",
                        help="Only generate training masks for all frames, no COLMAP re-run")
    parser.add_argument("--extract-videos", action="store_true",
                        help="Extract frames from cam*.mp4 into cam_XXX/YYYYY.png before processing")

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

    if args.extract_videos:
        extractvideos(videopath, startframe, endframe)

    if args.masks_only:
        print(f"Saving masks for frames {startframe}..{endframe-1}")
        for offset in tqdm.tqdm(range(startframe, endframe)):
            savemasks(videopath, offset)
        quit()

    if args.regen_images:
        print(f"Re-compositing {endframe - startframe} frame(s) over white background (no COLMAP re-run)")
        regeneratetrainimages(videopath, startframe, endframe)
        print("Done. Re-run training with white_background: true in your config.")
        quit()

    duration = endframe - startframe
    print(f"Preparing {duration} frame(s) [{startframe}..{endframe-1}]")
    print(f"  → set \"duration\": {duration} in your training config")

    use_poses_bounds = os.path.exists(os.path.join(videopath, "poses_bounds.npy"))
    if use_poses_bounds:
        print("found poses_bounds.npy — using ground-truth calibration (N3D-style pipeline)")
    else:
        print("no poses_bounds.npy — running full COLMAP SfM")

    # Frame 0: recover camera poses
    if startframe == 0:
        print("preparing colmap input for frame 0")
        preparecolmapframes(videopath, 0)
        if use_poses_bounds:
            print("running feature extraction + known-pose triangulation for frame 0")
            colmapfromposesbounds(videopath, 0)
        else:
            print("running full COLMAP SfM for frame 0 (feature extraction, matching, mapper, undistortion)")
            getcolmapsinglecustom(videopath, 0)
        print("saving training masks for frame 0")
        savemasks(videopath, 0)

    # Frames 1+: place images, then triangulate per-frame point cloud
    remaining = range(max(startframe, 1), endframe)
    if remaining:
        print(f"preparing images for frames {remaining.start}–{remaining.stop - 1}")
        for offset in tqdm.tqdm(remaining):
            prepareimagesonly(videopath, offset)
        print(f"triangulating per-frame point clouds for frames {remaining.start}–{remaining.stop - 1}")
        for offset in tqdm.tqdm(remaining):
            triangulateperframe(videopath, offset)
