#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import numpy as np 
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")
    parser.add_argument("--comp", action="store_true")
    parser.add_argument("--store_npz", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    args.save_iterations.append(args.iterations)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if args.model == 'stg':
        args.comp = False
        args.store_npz = False
        print("STG is not supported by post-processing or npz storage.")

    return args, lp.extract(args), op.extract(args), pp.extract(args)

def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]




def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v3")
    parser.add_argument("--valloader", type=str, default = "colmap")
    parser.add_argument("--configpath", type=str, default = "1")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # configpath
    safe_state(args.quiet)
    
    multiview = True if args.valloader.endswith("mv") else False

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
        print("args: " + str(args))
        
        return args, model.extract(args), pipeline.extract(args), multiview

def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)





def getcolmapsingleimundistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


 

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    #Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
   



def getcolmapsingleimdistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 
    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        

def getcolmapsingletechni(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    return


def getcolmapsinglecustom(folder, offset):
    """Full SfM pipeline for custom datasets with no known poses.
    Uses colmap mapper (not point_triangulator) to estimate intrinsics and
    extrinsics from scratch.
    """
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted", "sparse")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    maskfolder = os.path.join(folder, "input_masks")
    featureextract = ("colmap feature_extractor --database_path " + dbfile + " --image_path " + inputimagefolder
        + " --ImageReader.single_camera 1"
        + " --ImageReader.camera_model PINHOLE"
        + " --SiftExtraction.max_num_features 16384"
        + " --SiftExtraction.peak_threshold 0.001"
        + " --SiftExtraction.edge_threshold 20"
        + (" --ImageReader.mask_path " + maskfolder if (os.path.exists(maskfolder) and os.listdir(maskfolder)) else ""))
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = ("colmap exhaustive_matcher --database_path " + dbfile)
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

    # mapper estimates poses from scratch (no prior model needed)
    mapper = ("colmap mapper --database_path " + dbfile + " --image_path " + inputimagefolder
        + " --output_path " + distortedmodel
        + " --Mapper.ba_global_function_tolerance=0.000001"
        + " --Mapper.init_min_tri_angle 2"
        + " --Mapper.multiple_models 0"
        + " --Mapper.abs_pose_min_num_inliers 10"
        + " --Mapper.abs_pose_min_inlier_ratio 0.05"
        + " --Mapper.filter_min_tri_angle 0.5"
        + " --Mapper.tri_min_angle 0.5"
        + " --Mapper.filter_max_reproj_error 8"
        + " --Mapper.ba_refine_focal_length 0"
        + " --Mapper.ba_refine_extra_params 0")
    exit_code = os.system(mapper)
    if exit_code != 0:
        exit(exit_code)
    print(mapper)

    # mapper writes to distorted/sparse/0 — use that as input to undistorter.
    # COLMAP's image_undistorter CHECKs that its output images/ dir does not
    # already exist; wipe it (and sparse/) before every run so re-runs are safe.
    mapperout = os.path.join(distortedmodel, "0")
    for _d in ["images", "sparse", "stereo"]:
        _p = os.path.join(folder, _d)
        if os.path.exists(_p):
            shutil.rmtree(_p)
    img_undist_cmd = "colmap image_undistorter --image_path " + inputimagefolder \
        + " --input_path " + mapperout + " --output_path " + folder \
        + " --output_type COLMAP"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def triangulateperframe(basefolder, offset):
    """Triangulate a per-frame point cloud for frame `offset` using camera poses
    from colmap_0 (static rig). Gives each frame its own points3D.bin with
    geometry specific to that time step, which the training merges with timestamps.

    Requires colmap_0 to already be processed by getcolmapsinglecustom.
    """
    colmap0_sparse = os.path.join(basefolder, "colmap_0", "sparse", "0")
    folder = os.path.join(basefolder, "colmap_" + str(offset))
    images_dir = os.path.join(folder, "images")
    input_dir = os.path.join(folder, "input")
    mask_dir = os.path.join(folder, "input_masks")
    manual_dir = os.path.join(folder, "manual")
    tri_out = os.path.join(folder, "distorted", "sparse")
    sparse_dst = os.path.join(folder, "sparse", "0")
    dbfile = os.path.join(folder, "input.db")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(manual_dir, exist_ok=True)
    os.makedirs(tri_out, exist_ok=True)
    os.makedirs(sparse_dst, exist_ok=True)

    # Use colmap_0 binary sparse directly — no TXT conversion needed, and binary
    # preserves rig metadata that TXT export drops.

    # Copy undistorted images to input/ for feature extraction
    for img in sorted(os.listdir(images_dir)):
        if not img.endswith(".png"):
            continue
        src = os.path.join(images_dir, img)
        dst = os.path.join(input_dir, img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Seed the DB from colmap_0 to preserve camera/rig definitions (COLMAP 3.10+
    # uses a rig/frame schema; a fresh DB gets different RigIds than colmap_0's
    # reconstruction, causing point_triangulator to crash with RigId mismatch).
    colmap0_db = os.path.join(basefolder, "colmap_0", "input.db")
    if os.path.exists(colmap0_db):
        shutil.copy(colmap0_db, dbfile)
        import sqlite3
        conn = sqlite3.connect(dbfile)
        for tbl in ["keypoints", "descriptors", "matches", "two_view_geometries", "images", "frames"]:
            try:
                conn.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass
        conn.commit()
        conn.close()
    elif os.path.exists(dbfile):
        os.remove(dbfile)

    # Feature extraction — reuse camera 1 from colmap_0 (no single_camera flag
    # needed since the camera already exists in the seeded DB)
    cmd = ("colmap feature_extractor"
           " --database_path " + dbfile +
           " --image_path " + input_dir +
           " --ImageReader.existing_camera_id 1"
           " --ImageReader.camera_model PINHOLE"
           " --SiftExtraction.max_num_features 16384"
           " --SiftExtraction.peak_threshold 0.001"
           " --SiftExtraction.edge_threshold 20"
           + (" --ImageReader.mask_path " + mask_dir if (os.path.exists(mask_dir) and os.listdir(mask_dir)) else ""))
    exit_code = os.system(cmd)
    if exit_code != 0:
        exit(exit_code)

    # Feature matching
    cmd = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(cmd)
    if exit_code != 0:
        exit(exit_code)

    # Triangulate using known poses from colmap_0 (binary, preserves rig info)
    cmd = ("colmap point_triangulator"
           " --database_path " + dbfile +
           " --image_path " + input_dir +
           " --output_path " + tri_out +
           " --input_path " + colmap0_sparse +
           " --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(cmd)
    if exit_code != 0:
        exit(exit_code)

    # Copy resulting points3D.bin to sparse/0/
    src = os.path.join(tri_out, "points3D.bin")
    dst = os.path.join(sparse_dst, "points3D.bin")
    if os.path.exists(src):
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    else:
        print(f"warning: triangulation produced no points3D.bin for frame {offset}")

    # Cleanup input images
    shutil.rmtree(input_dir, ignore_errors=True)
