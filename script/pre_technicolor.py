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
import sys
import argparse
from PIL import Image

sys.path.append(".")




    


def updatetechnicamerasindb(dbfile, videopath, manualfolder):
    """After feature_extractor, update cameras with Technicolor intrinsics and write manual/ files.
    Uses positional matching: DB images sorted by name map 1-to-1 to cameras_parameters.txt rows."""
    camparam_path = os.path.join(videopath, "cameras_parameters.txt")
    camparams = []
    with open(camparam_path, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for lineidx, row in enumerate(reader):
            if lineidx == 0:
                continue  # skip header line
            row = [float(c) for c in row if c.strip() != '']
            camparams.append(row)

    con = sqlite3.connect(dbfile)
    rows = con.execute("SELECT image_id, name, camera_id FROM images ORDER BY name").fetchall()

    imagetxtlist = []
    cameratxtlist = []
    W, H = 2048, 1088

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


def getcolmapsingletechni_v2(videopath, offset):
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
    updatetechnicamerasindb(dbfile, videopath, manualfolder)

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
    scenenamelist = ["Train"]
    framerangedict = {}
    framerangedict["Birthday"] = [_ for _ in range(151, 201)] # start from 1
    framerangedict["Fabien"] = [_ for _ in range(51, 101)] # start from 1
    framerangedict["Painter"] = [_ for _ in range(100, 150)] # start from 0
    framerangedict["Theater"] = [_ for _ in range(51, 101)] # start from 1
    framerangedict["Train"] = [_ for _ in range(151, 201)] # start from 1
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--videopath", default="", type=str)
    args = parser.parse_args()

    


    videopath = args.videopath

    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    srcscene = videopath.split("/")[-2]
    srcscene = srcscene[0].upper() + srcscene[1:]  # normalize to Title case
    print("srcscene", srcscene)

    if srcscene == "Birthday":
        print("check broken")
        fixbroken(videopath + "Birthday_undist_00173_09.png", videopath + "Birthday_undist_00172_09.png")
        
    imagecopy(videopath, offsetlist=framerangedict[srcscene])

    for offset in tqdm.tqdm(range(0, 50)):
        getcolmapsingletechni_v2(videopath, offset=offset)

    #  rm -r colmap_* # once meet error, delete all colmap_* folders and rerun this script.


