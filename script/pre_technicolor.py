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

import os
import cv2
import glob
import tqdm
import numpy as np
import shutil
import pickle
import sqlite3

import natsort
import struct
import pickle
import csv
import json
import sys
import argparse
from PIL import Image

sys.path.append(".")




    


def _parse_camparams(camparam_path):
    """Parse cameras_parameters.txt, returning a list of float-column rows.
    Tolerates an optional leading non-numeric column (e.g. cam_name written by
    flamsplat) and skips comment / header lines.

    A valid data row must have at least 12 numeric columns:
    fx cx cy _ _ qw qx qy qz tx ty tz. Rows shorter than that are treated as
    headers and skipped (e.g. the original Technicolor 5-col header line).
    """
    out = []
    with open(camparam_path, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            cells = [c for c in row if c.strip() != ""]
            if not cells:
                continue
            if cells[0].startswith("#"):
                continue
            # If the first cell isn't a number (e.g. 'cam001'), drop it.
            try:
                float(cells[0])
            except ValueError:
                cells = cells[1:]
            try:
                nums = [float(c) for c in cells]
            except ValueError:
                continue
            if len(nums) < 12:
                # Header row (e.g. Technicolor's 5-col first line); skip.
                continue
            out.append(nums)
    return out


def updatetechnicamerasindb(dbfile, videopath, manualfolder, width=None, height=None):
    """After feature_extractor, update cameras with Technicolor intrinsics and write manual/ files.
    Uses positional matching: DB images sorted by name map 1-to-1 to cameras_parameters.txt rows.

    If width/height are not given, falls back to scene_meta.json, then to the original
    Technicolor hard-coded resolution (2048x1088)."""
    camparam_path = os.path.join(videopath, "cameras_parameters.txt")
    camparams = _parse_camparams(camparam_path)

    con = sqlite3.connect(dbfile)
    rows = con.execute("SELECT image_id, name, camera_id FROM images ORDER BY name").fetchall()

    imagetxtlist = []
    cameratxtlist = []

    # Resolve W,H precedence: caller arg > scene_meta.json > legacy default.
    if width is None or height is None:
        meta_path = os.path.join(videopath, "scene_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if width is None:
                width = int(meta["width"])
            if height is None:
                height = int(meta["height"])
    if width is None or height is None:
        width, height = 2048, 1088
    W, H = int(width), int(height)

    for db_idx, (image_id, imgname, camera_id) in enumerate(rows):
        if db_idx >= len(camparams):
            print(f"WARNING: no camera param entry for DB image idx {db_idx} ({imgname}), skipping")
            continue
        p = camparams[db_idx]
        fx = p[0]
        cx = p[1]
        cy = p[2]
        colmapQ = [p[5], p[6], p[7], p[8]]  # qw qx qy qz
        colmapT = [p[9], p[10], p[11]]

        params = np.array([fx, fx, cx, cy], dtype=np.float64)
        con.execute(
            "UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (W, H, params.tobytes(), camera_id)
        )

        line = (str(image_id) + " " +
                " ".join(str(v) for v in colmapQ) + " " +
                " ".join(str(v) for v in colmapT) + " " +
                str(camera_id) + " " + imgname + "\n")
        imagetxtlist.append(line)
        imagetxtlist.append("\n")
        cameratxtlist.append(
            f"{camera_id} PINHOLE {W} {H} {fx} {fx} {cx} {cy}\n"
        )

    con.commit()
    con.close()

    os.makedirs(manualfolder, exist_ok=True)
    with open(os.path.join(manualfolder, "images.txt"), "w") as f:
        f.writelines(imagetxtlist)
    with open(os.path.join(manualfolder, "cameras.txt"), "w") as f:
        f.writelines(cameratxtlist)
    with open(os.path.join(manualfolder, "points3D.txt"), "w") as f:
        pass


def getcolmapsingletechni_v2(videopath, offset, width=None, height=None):
    """COLMAP 3.13-compatible pipeline for Technicolor datasets.
    delete DB → feature_extractor → update cameras + write manual/ → exhaustive_matcher
    → point_triangulator → image_undistorter → move sparse/0."""
    folder = os.path.join(videopath, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    manualfolder = os.path.join(folder, "manual")
    distortedmodel = os.path.join(folder, "distorted/sparse")

    os.makedirs(distortedmodel, exist_ok=True)

    # Delete DB so feature_extractor creates fresh COLMAP 3.x schema
    if os.path.exists(dbfile):
        os.remove(dbfile)

    featureextract = f"colmap feature_extractor --database_path {dbfile} --image_path {inputimagefolder}"
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    # Update cameras with known intrinsics + write manual/ with actual DB image IDs
    updatetechnicamerasindb(dbfile, videopath, manualfolder, width=width, height=height)

    featurematcher = f"colmap exhaustive_matcher --database_path {dbfile}"
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

    triandmap = (f"colmap point_triangulator --database_path {dbfile}"
                 f" --image_path {inputimagefolder}"
                 f" --output_path {distortedmodel}"
                 f" --input_path {manualfolder}"
                 f" --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(triandmap)
    if exit_code != 0:
        exit(exit_code)

    img_undist_cmd = (f"colmap image_undistorter"
                      f" --image_path {inputimagefolder}"
                      f" --input_path {distortedmodel}"
                      f" --output_path {folder}"
                      f" --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(os.path.join(folder, "sparse"))
    os.makedirs(os.path.join(folder, "sparse", "0"), exist_ok=True)
    for file in files:
        if file == '0':
            continue
        shutil.move(os.path.join(folder, "sparse", file),
                    os.path.join(folder, "sparse", "0", file))












RGB_EXTS = (".png", ".jpg", ".jpeg")


def imagecopy_from_flamsplat(base_dir, frame_indices):
    """Map flamsplat's per-camera layout into Technicolor's per-frame layout.

    Source:  base_dir/cam_NNN/rgb/FFFFF.{png,jpg,jpeg}  and  base_dir/cam_NNN/depth/FFFFF.exr
    Target:  base_dir/colmap_<idx>/input/camNNN.<ext>   and  base_dir/colmap_<idx>/depth/camNNN.exr

    The output image extension is preserved from the source (so PNG stays PNG,
    JPEG stays JPEG). COLMAP and the Technicolor reader both handle either.

    `frame_indices` is the list of source frame indices to use, one per output
    `colmap_<idx>` (idx = position in the list). The cam_NNN -> camNNN rename
    drops the underscore so that the substring "cam10" no longer aliases cam100..109
    in the Technicolor reader's holdout logic.
    """
    cam_dirs = sorted(glob.glob(os.path.join(base_dir, "cam_*")))
    if not cam_dirs:
        raise RuntimeError(f"No cam_* directories found in {base_dir}")

    for idx, frame in enumerate(frame_indices):
        target_input = os.path.join(base_dir, f"colmap_{idx}", "input")
        target_depth = os.path.join(base_dir, f"colmap_{idx}", "depth")
        os.makedirs(target_input, exist_ok=True)
        os.makedirs(target_depth, exist_ok=True)

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir.rstrip("/"))  # cam_NNN
            try:
                cam_num = int(cam_name.split("_")[1])
            except (IndexError, ValueError):
                # Skip non-cam dirs that might match cam_*
                continue
            new_name = f"cam{cam_num:03d}"

            # RGB — try PNG / JPG / JPEG with several pad widths; preserve ext.
            rgb_src = None
            for pad in (5, 4, 3, 2, 1):
                stem = str(frame).zfill(pad)
                for ext in RGB_EXTS:
                    cand = os.path.join(cam_dir, "rgb", stem + ext)
                    if os.path.exists(cand):
                        rgb_src = cand
                        break
                if rgb_src is not None:
                    break
            if rgb_src is None:
                raise FileNotFoundError(
                    f"No RGB ({'/'.join(RGB_EXTS)}) for frame {frame} in {cam_dir}/rgb/"
                )
            src_ext = os.path.splitext(rgb_src)[1]
            shutil.copy(rgb_src, os.path.join(target_input, new_name + src_ext))

            # Depth (optional, legacy — older flamsplat scenes)
            depth_dir = os.path.join(cam_dir, "depth")
            if os.path.isdir(depth_dir):
                for pad in (5, 4, 3, 2, 1):
                    for ext in (".exr", ".png", ".jpg"):
                        cand = os.path.join(depth_dir, str(frame).zfill(pad) + ext)
                        if os.path.exists(cand):
                            shutil.copy(
                                cand, os.path.join(target_depth, new_name + ext)
                            )
                            break
                    else:
                        continue
                    break

            # Mask (optional, used by alpha-loss training).
            # Mirrors the depth block: walk pad widths × extensions, copy with
            # the camNNN basename so it sits next to colmap_<idx>/images/camNNN.<ext>.
            mask_dir_src = os.path.join(cam_dir, "mask")
            if os.path.isdir(mask_dir_src):
                target_mask_dir = os.path.join(base_dir, f"colmap_{idx}", "mask")
                os.makedirs(target_mask_dir, exist_ok=True)
                for pad in (5, 4, 3, 2, 1):
                    stem = str(frame).zfill(pad)
                    for ext in (".png", ".jpg", ".jpeg"):
                        cand = os.path.join(mask_dir_src, stem + ext)
                        if os.path.exists(cand):
                            shutil.copy(
                                cand, os.path.join(target_mask_dir, new_name + ext)
                            )
                            break
                    else:
                        continue
                    break


def imagecopy(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    import cv2
    import numpy as np
    import os 
    import json 
    
    pnglist = glob.glob(video + "/*.png")

    for pngpath in pnglist:
        pass 
    
    for idx , offset in enumerate(offsetlist):
        pnglist = glob.glob(video + "*_undist_" + str(offset).zfill(5)+"_*.png")
        
        targetfolder = os.path.join(video, "colmap_" + str(idx), "input")
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for pngpath in pnglist:
            cameraname = os.path.basename(pngpath).split("_")[3]
            newpath = os.path.join(targetfolder, "cam" + cameraname )
            shutil.copy(pngpath, newpath)
    





def checkimage(videopath):
    from PIL import Image

    import cv2
    imagelist = glob.glob(videopath + "*.png")
    for imagepath in imagelist:
        try:
            img = Image.open(imagepath) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
                print('Bad file:', imagepath) # print out the names of corrupt files
        bad_file_list=[]
        bad_count=0
        try:
            img.cv2.imread(imagepath)
            shape=img.shape # this will throw an error if the img is not read correctly
        except:
            bad_file_list.append(imagepath)
            bad_count +=1
    print(bad_file_list)

def fixbroken(imagepath, refimagepath):
    try:
        img = Image.open(imagepath) # open the image file
        print("start verifying", imagepath)
        img.verify() # if we already fixed it. 
        print("already fixed", imagepath)
    except :
        print('Bad file:', imagepath)
        import cv2
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(imagepath)
        
        img.load()
        img.save("tmp.png")

        savedimage = cv2.imread("tmp.png")
        mask = savedimage == 0
        refimage = cv2.imread(refimagepath)
        composed = savedimage * (1-mask) + refimage * (mask)
        cv2.imwrite(imagepath, composed)
        print("fixing done", imagepath)
        os.remove("tmp.png")


if __name__ == "__main__" :
    # Hard-coded ranges used by the original Technicolor scenes.
    framerangedict = {
        "Birthday": list(range(151, 201)),
        "Fabien":   list(range(51, 101)),
        "Painter":  list(range(100, 150)),
        "Theater":  list(range(51, 101)),
        "Train":    list(range(151, 201)),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", required=True, type=str)
    parser.add_argument(
        "--format",
        choices=("technicolor", "flamsplat"),
        default="technicolor",
        help="technicolor: original Technicolor scenes (flat *_undist_FFFFF_CC.png at root). "
             "flamsplat: synthetic scenes rendered by script/flamsplat.py "
             "(per-camera cam_NNN/rgb/FFFFF.png + cam_NNN/depth/FFFFF.exr).",
    )
    parser.add_argument("--width", type=int, default=None,
                        help="Image width in pixels. If unset, read from scene_meta.json.")
    parser.add_argument("--height", type=int, default=None,
                        help="Image height in pixels. If unset, read from scene_meta.json.")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="flamsplat only: number of frames to process (overrides scene_meta.json).")
    parser.add_argument("--start_offset", type=int, default=0,
                        help="flamsplat only: source frame index of the first colmap_0 (default 0).")
    args = parser.parse_args()

    videopath = args.videopath
    if not videopath.endswith("/"):
        videopath = videopath + "/"

    # Resolve width/height from scene_meta.json if available (flamsplat writes it).
    width, height = args.width, args.height
    meta_path = os.path.join(videopath, "scene_meta.json")
    if (width is None or height is None) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if width is None:
            width = int(meta["width"])
        if height is None:
            height = int(meta["height"])

    if args.format == "flamsplat":
        # Determine frame indices
        num_frames = args.num_frames
        if num_frames is None and os.path.exists(meta_path):
            with open(meta_path) as f:
                num_frames = int(json.load(f)["frame_count"])
        if num_frames is None:
            raise SystemExit("--num_frames required (or write scene_meta.json with frame_count)")
        frame_indices = [args.start_offset + i for i in range(num_frames)]

        print(f"flamsplat mode: {num_frames} frames, {width}x{height}")
        imagecopy_from_flamsplat(videopath, frame_indices)
        for offset in tqdm.tqdm(range(num_frames)):
            getcolmapsingletechni_v2(videopath, offset=offset, width=width, height=height)
    else:
        srcscene = videopath.split("/")[-2]
        srcscene = srcscene[0].upper() + srcscene[1:]
        print("srcscene", srcscene)

        if srcscene == "Birthday":
            print("check broken")
            fixbroken(videopath + "Birthday_undist_00173_09.png",
                      videopath + "Birthday_undist_00172_09.png")

        imagecopy(videopath, offsetlist=framerangedict[srcscene])
        for offset in tqdm.tqdm(range(0, 50)):
            getcolmapsingletechni_v2(videopath, offset=offset, width=width, height=height)

    #  rm -r colmap_* # once meet error, delete all colmap_* folders and rerun this script.


