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
import sys 
import argparse
sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")




def extractframes(videopath, startframe=0, endframe=300, fmt="jpg", quality=95):
    outdir = videopath.replace(".mp4", "")
    needed = list(range(startframe, endframe))
    if all(os.path.exists(os.path.join(outdir, str(i) + "." + fmt)) for i in needed):
        print("already extracted needed frames, skip extracting")
        return
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
    cam = cv2.VideoCapture(videopath)
    for i in range(endframe):
        success, frame = cam.read()
        if not success:
            print(f"error reading frame {i}")
            break
        if i >= startframe:
            cv2.imwrite(os.path.join(outdir, str(i) + "." + fmt), frame, params)
    cam.release()


def resolveframepath(camfolder, frame_idx):
    """Find the image for frame_idx, supporting plain (0.ext) or zero-padded (00000.ext) naming."""
    names = [str(frame_idx)] + [str(frame_idx).zfill(pad) for pad in range(2, 7)]
    for name in names:
        for ext in SUPPORTED_EXTS:
            path = os.path.join(camfolder, name + ext)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"No image for frame {frame_idx} in {camfolder}")


def preparecolmapdynerf(folder, offset=0):
    folderlist = sorted(glob.glob(folder + "cam**/"))
    savedir = os.path.join(folder, "colmap_" + str(offset))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for camfolder in folderlist:
        imagepath = resolveframepath(camfolder, offset)
        src_ext = os.path.splitext(imagepath)[1]
        imagesavepath = os.path.join(savedir, camfolder.split("/")[-2] + src_ext)
        shutil.copy(imagepath, imagesavepath)


def getcampaths(path):
    """Return sorted list of (name, source) tuples for cameras, from videos or image folders."""
    video_paths = sorted(glob.glob(os.path.join(path, 'cam*.mp4')))
    if video_paths:
        return video_paths, "video"
    cam_folders = sorted(glob.glob(os.path.join(path, 'cam*/')))
    if cam_folders:
        return cam_folders, "images"
    raise RuntimeError(f"No cam*.mp4 files or cam*/ folders found in {path}")


def convertdynerftocolmapdb(path, offset=0):
    originnumpy = os.path.join(path, "poses_bounds.npy")
    cam_sources, source_type = getcampaths(path)
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()

    # Detect image extension from files in input/
    input_dir = os.path.join(projectfolder, "input")
    sample = next((f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))), None)
    img_ext = os.path.splitext(sample)[1] if sample else ".png"

    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1,2,0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)


        for i in range(len(poses)):
            src = cam_sources[i]
            if source_type == "video":
                cameraname = os.path.basename(src)[:-4]  # strip .mp4
            else:
                cameraname = os.path.basename(src.rstrip("/"))
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i+1)
            cameraid = imageid
            pngname = cameraname + img_ext
            
            line =  imageid + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(T[j]) + " "
            line = line  + cameraid + " " + pngname + "\n"
            empltyline = "\n"
            imagetxtlist.append(line)
            imagetxtlist.append(empltyline)

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
            cameratxtlist.append(cameraline)
            
            image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
            db.commit()
        db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 





if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)
    parser.add_argument("--format", default="jpg", choices=["png", "jpg"], help="Output image format (default: jpg)")
    parser.add_argument("--jpeg-quality", default=95, type=int, help="JPEG quality 1-100 (default: 95)")

    args = parser.parse_args()
    videopath = args.videopath
    fmt = args.format
    quality = args.jpeg_quality

    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    
    
    ##### step1
    videoslist = glob.glob(videopath + "*.mp4")
    if videoslist:
        print(f"start extracting frames {startframe}-{endframe} from videos")
        for v in tqdm.tqdm(videoslist):
            extractframes(v, startframe, endframe, fmt=fmt, quality=quality)
    else:
        print("no videos found, assuming images are already extracted in cam*/ folders")

    # # ## step2 prepare colmap input
    print("start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmapdynerf(videopath, offset)


    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file 
    for offset in range(startframe, endframe):
        convertdynerftocolmapdb(videopath, offset)


    # ## step 4 run colmap, per frame, if error, reinstall opencv-headless 
    for offset in range(startframe, endframe):
        getcolmapsinglen3d(videopath, offset)