#!/usr/bin/env python3
"""post_blender.py — bypass pre_technicolor.py with native synthetic data.

Workflow:

    flamsplat.py (Blender, renders + exports cameras_parameters.txt + pointsamples)
        |
        v
    post_blender.py  <-- you are here
        |
        v
    train.py with --source_path <scene>/colmap_0 (loader=technicolor)

What it produces, per scene folder:

    scene/
        cameras_parameters.txt    (already there, from flamsplat)
        scene_meta.json           (already there)
        cam_NNN/rgb/FFFFF.png     (already there, rendered RGB)
        cam_NNN/mask/FFFFF.png    (already there, optional)
        pointsamples/FFFFF.npz    (already there, from WM_OT_ExportPointSamples)

        colmap_0/
            images/cam001.png, cam002.png, ...        (copied from cam_NNN/rgb/00000.*)
            mask/cam001.png, ...                       (copied from cam_NNN/mask/00000.*, optional)
            sparse/0/cameras.bin     <-- shared intrinsics
            sparse/0/images.bin      <-- shared extrinsics (one per camera)
            sparse/0/points3D.bin    <-- frame-0 point cloud, colored

        colmap_1/
            images/cam001.png, ...                     (copied from cam_NNN/rgb/00001.*)
            sparse/0/points3D.bin    <-- frame-1 point cloud, colored
            (no cameras.bin / images.bin — trainer only reads colmap_<starttime>'s)
        ...
        colmap_<frame_count-1>/
            ...

The trainer (cstg/thirdparty/gaussian_splatting/scene/dataset_readers.py
::readColmapSceneInfoTechnicolor) reads cameras.bin/images.bin only from
the starttime folder (parsed from the source path) and walks
colmap_<starttime>..colmap_<starttime+duration-1>/sparse/0/points3D.bin to
build a spatiotemporal init cloud. We therefore write the bin files only
under colmap_0/ (assuming starttime=0) — saves ~50x duplicated metadata.

Coordinate conventions are inherited from flamsplat.py's Technicolor
exporter: cameras_parameters.txt lists per-camera
    cam_name fx cx cy _ _ qw qx qy qz tx ty tz
where (qw, qx, qy, qz, tx, ty, tz) is the world->camera (COLMAP) extrinsic
and the world frame is identical to Blender's world frame. We project
points in that same world frame.

Usage:
    python post_blender.py --scene_path /path/to/flamsplat_scene
                           [--starttime 0]
                           [--num_frames N]    # default: from scene_meta.json
                           [--copy_images]    # copy RGB into colmap_<i>/images/
                                              # (default on; use --no_copy_images
                                              #  to symlink instead — saves disk)
                           [--copy_masks]     # also copy masks if present
                           [--max_points_per_frame N]  # cap per-frame points
"""

import argparse
import csv
import glob
import json
import os
import shutil
import struct
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# COLMAP binary writers (matches scene/colmap_loader.py's readers)
# ---------------------------------------------------------------------------

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
}


def write_cameras_bin(path: str, cameras: List[dict]) -> None:
    """Write a COLMAP cameras.bin.

    `cameras` is a list of dicts with keys:
        camera_id (int), model (str: 'PINHOLE'), width (int), height (int),
        params (list of float of length 4 for PINHOLE: [fx, fy, cx, cy]).
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            model_id = CAMERA_MODEL_IDS[cam["model"]]
            f.write(struct.pack("<iiQQ",
                                int(cam["camera_id"]),
                                int(model_id),
                                int(cam["width"]),
                                int(cam["height"])))
            for p in cam["params"]:
                f.write(struct.pack("<d", float(p)))


def write_images_bin(path: str, images: List[dict]) -> None:
    """Write a COLMAP images.bin (with empty 2D-point tracks).

    `images` is a list of dicts with keys:
        image_id (int), qvec (length-4: qw,qx,qy,qz), tvec (length-3),
        camera_id (int), name (str).
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", int(img["image_id"])))
            for q in img["qvec"]:
                f.write(struct.pack("<d", float(q)))
            for t in img["tvec"]:
                f.write(struct.pack("<d", float(t)))
            f.write(struct.pack("<I", int(img["camera_id"])))
            name = img["name"]
            f.write(name.encode("utf-8"))
            f.write(b"\x00")
            # num_points2D = 0 (no track info; reader doesn't use it)
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a COLMAP points3D.bin with empty track lists.

    `xyz` is (N, 3) float; `rgb` is (N, 3) uint8.

    The trainer's read_points3D_binary reads (point_id, x, y, z, R, G, B,
    error, track_length, track_entries...). It uses XYZ + RGB; track and
    error are read but discarded. We write track_length=0 and error=0.
    """
    n = int(xyz.shape[0])
    assert rgb.shape == (n, 3), f"rgb shape {rgb.shape} != ({n}, 3)"
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack(
                "<QdddBBBd",
                i + 1,  # point3D_id, 1-indexed
                float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]),
                0.0,  # reprojection error
            ))
            # track_length = 0  -> no track entries follow
            f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Camera-parameter parsing (matches flamsplat.py's Technicolor export)
# ---------------------------------------------------------------------------

def parse_cameras_parameters(path: str) -> List[dict]:
    """Parse cameras_parameters.txt as written by flamsplat's
    WM_OT_ExportSplatTechnicolor:

        # name fx cx cy _ _ qw qx qy qz tx ty tz
        cam000 fx cx cy 0.0 0.0 qw qx qy qz tx ty tz
        cam001 ...

    Returns list of dicts with keys: name, fx, cx, cy, qvec (4,), tvec (3,).
    The qvec/tvec are world->camera (COLMAP convention).
    """
    out = []
    with open(path, "r") as f:
        for row in csv.reader(f, delimiter=" "):
            cells = [c for c in row if c.strip() != ""]
            if not cells or cells[0].startswith("#"):
                continue
            # First cell is the cam name (non-numeric); rest are floats.
            try:
                float(cells[0])
                # No name column — shouldn't happen with flamsplat's writer,
                # but handle gracefully by synthesizing.
                name = None
                nums = [float(c) for c in cells]
            except ValueError:
                name = cells[0]
                nums = [float(c) for c in cells[1:]]
            if len(nums) < 12:
                continue
            fx = nums[0]
            cx = nums[1]
            cy = nums[2]
            qvec = np.array(nums[5:9], dtype=np.float64)   # qw qx qy qz
            tvec = np.array(nums[9:12], dtype=np.float64)
            if name is None:
                name = f"cam{len(out):03d}"
            out.append(dict(name=name, fx=fx, cx=cx, cy=cy, qvec=qvec, tvec=tvec))
    return out


def qvec_to_R(qvec: np.ndarray) -> np.ndarray:
    """COLMAP convention quaternion (qw, qx, qy, qz) -> 3x3 rotation."""
    qw, qx, qy, qz = qvec
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),     s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw),     1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw),     s * (qy * qz + qx * qw),     1 - s * (qx * qx + qy * qy)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Projection + color sampling
# ---------------------------------------------------------------------------

def project_points_to_cameras(
    xyz_world: np.ndarray,         # (N, 3) world-space points
    cameras: List[dict],           # parsed cameras_parameters.txt rows
    width: int,
    height: int,
    normal_world: np.ndarray = None,  # (N, 3) optional surface normals
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point, find the best camera that sees it and return:
        cam_idx     (N,) int    -- index into `cameras`, -1 if no camera sees it
        pixel_uv    (N, 2) int  -- (u, v) pixel coords in that camera (0 if -1)
        valid       (N,) bool   -- whether the point was assigned to a camera

    "Best" criterion: of all cameras where the point projects inside the
    image and lies in front of the camera, pick the one with smallest
    depth (closest, least likely to be self-occluded behind something).
    If `normal_world` is given, also require the point to face the camera
    (dot(normal, view_dir_to_cam) > 0) — backfaces are skipped.
    """
    n = xyz_world.shape[0]
    best_cam = np.full(n, -1, dtype=np.int32)
    best_uv = np.zeros((n, 2), dtype=np.int32)
    best_depth = np.full(n, np.inf, dtype=np.float64)

    for ci, cam in enumerate(cameras):
        R = qvec_to_R(cam["qvec"])     # world -> camera rotation
        t = cam["tvec"]                # world -> camera translation
        # COLMAP camera convention: cam frame is +Z forward (looking down +Z)
        cam_xyz = xyz_world @ R.T + t  # (N, 3)
        z = cam_xyz[:, 2]
        in_front = z > 1e-6

        fx, cy_p, cx_p = cam["fx"], cam["cy"], cam["cx"]
        # flamsplat writes fx==fy (single focal). Pixel projection:
        u = (fx * cam_xyz[:, 0] / np.where(in_front, z, 1.0)) + cx_p
        v = (fx * cam_xyz[:, 1] / np.where(in_front, z, 1.0)) + cy_p
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height) & in_front

        if normal_world is not None:
            # Camera center in world: C = -R^T @ t
            cam_center_world = -R.T @ t
            view_dir = cam_center_world[None, :] - xyz_world   # (N, 3), points to cam
            view_dir_norm = np.linalg.norm(view_dir, axis=1, keepdims=True)
            view_dir_norm = np.where(view_dir_norm > 1e-12, view_dir_norm, 1.0)
            view_unit = view_dir / view_dir_norm
            facing = np.sum(normal_world * view_unit, axis=1) > 0.1
            in_image &= facing

        # Update best (closest) cam for each in-image point
        improved = in_image & (z < best_depth)
        best_cam[improved] = ci
        best_uv[improved, 0] = np.clip(u[improved], 0, width - 1).astype(np.int32)
        best_uv[improved, 1] = np.clip(v[improved], 0, height - 1).astype(np.int32)
        best_depth[improved] = z[improved]

    valid = best_cam >= 0
    return best_cam, best_uv, valid


def sample_colors_from_images(
    cam_idx: np.ndarray,         # (N,) int
    pixel_uv: np.ndarray,         # (N, 2) int
    valid: np.ndarray,            # (N,) bool
    image_paths: List[str],       # one per cam index
) -> np.ndarray:
    """For each valid point, look up the pixel color in its assigned camera's
    rendered image. Invalid points get gray (128,128,128).

    Caches each image once via PIL -> np.array, so this is one read per cam.
    """
    n = cam_idx.shape[0]
    rgb = np.full((n, 3), 128, dtype=np.uint8)

    if not np.any(valid):
        return rgb

    # Cache images we actually need
    used_cams = np.unique(cam_idx[valid])
    cache = {}
    for ci in used_cams:
        path = image_paths[int(ci)]
        img = Image.open(path).convert("RGB")
        cache[int(ci)] = np.asarray(img, dtype=np.uint8)   # (H, W, 3)

    # Vectorized lookup per camera
    for ci_int in used_cams:
        ci_int = int(ci_int)
        mask = (cam_idx == ci_int) & valid
        if not np.any(mask):
            continue
        img_arr = cache[ci_int]
        u = pixel_uv[mask, 0]
        v = pixel_uv[mask, 1]
        rgb[mask] = img_arr[v, u]

    return rgb


# ---------------------------------------------------------------------------
# File-layout helpers (mirrors pre_technicolor.imagecopy_from_flamsplat)
# ---------------------------------------------------------------------------

RGB_EXTS = (".png", ".jpg", ".jpeg")


def find_frame_file(cam_dir: str, subdir: str, frame_idx: int) -> str:
    """Look up cam_dir/<subdir>/<padded>.<ext> with the various pad widths
    flamsplat might use. Returns the first hit or None."""
    for pad in (5, 4, 3, 2, 1):
        stem = str(frame_idx).zfill(pad)
        for ext in (".png", ".jpg", ".jpeg", ".exr"):
            cand = os.path.join(cam_dir, subdir, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def collect_per_frame_image_paths(
    scene_path: str, num_frames: int
) -> Tuple[List[List[str]], List[str], List[str]]:
    """Walk scene_path/cam_*/rgb/*.png and build, per frame, the list of
    per-camera image paths.

    Returns:
        per_frame_rgb : List[List[str]]   # [frame][cam_idx] -> rgb path
        per_frame_mask: List[List[str]]   # [frame][cam_idx] -> mask path or None
        cam_dirs      : List[str]          # sorted cam_NNN dirs (one per cam)
    """
    cam_dirs = sorted(glob.glob(os.path.join(scene_path, "cam_*")))
    if not cam_dirs:
        raise SystemExit(f"No cam_* dirs found in {scene_path}")

    per_frame_rgb = []
    per_frame_mask = []
    for f in range(num_frames):
        rgbs = []
        masks = []
        for cd in cam_dirs:
            rgb_p = find_frame_file(cd, "rgb", f)
            if rgb_p is None:
                raise SystemExit(
                    f"Missing RGB for frame {f} in {cd}/rgb/. "
                    f"Did you render frames before running post_blender?"
                )
            rgbs.append(rgb_p)
            m_p = find_frame_file(cd, "mask", f)
            masks.append(m_p)
        per_frame_rgb.append(rgbs)
        per_frame_mask.append(masks)
    return per_frame_rgb, per_frame_mask, cam_dirs


def materialize_images_dir(
    out_images_dir: str,
    rgb_paths: List[str],         # per-camera source paths for this frame
    cam_names: List[str],         # ['cam000', 'cam001', ...]
    mode: str = "copy",           # 'copy' or 'symlink'
) -> List[str]:
    """Copy/symlink each cam_NNN/rgb/FFFFF.<ext> to out_images_dir/camNNN.<ext>
    Returns the list of basenames written (matches order of cam_names).
    """
    os.makedirs(out_images_dir, exist_ok=True)
    basenames = []
    for src, name in zip(rgb_paths, cam_names):
        ext = os.path.splitext(src)[1]
        dst = os.path.join(out_images_dir, name + ext)
        # Overwrite any stale file
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy(src, dst)
        basenames.append(name + ext)
    return basenames


def materialize_masks_dir(
    out_mask_dir: str,
    mask_paths: List[str],        # may contain None for cams without masks
    cam_names: List[str],
    mode: str = "copy",
) -> int:
    """Copy/symlink masks. Returns the number actually written (skips Nones)."""
    os.makedirs(out_mask_dir, exist_ok=True)
    n_done = 0
    for src, name in zip(mask_paths, cam_names):
        if src is None:
            continue
        ext = os.path.splitext(src)[1]
        dst = os.path.join(out_mask_dir, name + ext)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy(src, dst)
        n_done += 1
    return n_done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_path", required=True,
                    help="Flamsplat scene root containing cameras_parameters.txt, "
                         "scene_meta.json, cam_*/rgb/, and pointsamples/.")
    ap.add_argument("--starttime", type=int, default=0,
                    help="Index of first colmap_<idx> folder to write. Default 0.")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="How many frames to materialize. Default: from scene_meta.json.")
    ap.add_argument("--no_copy_images", action="store_true",
                    help="Symlink images instead of copying (saves disk; needs filesystem "
                         "support for symlinks).")
    ap.add_argument("--no_masks", action="store_true",
                    help="Skip mask materialization even if cam_NNN/mask/ exists.")
    ap.add_argument("--max_points_per_frame", type=int, default=None,
                    help="Cap per-frame point count after filtering (random subsample).")
    ap.add_argument("--require_facing", action="store_true",
                    help="When picking the best camera per point, require the surface "
                         "normal to face the camera. Off by default since synthetic "
                         "scenes often have one-sided geometry.")
    args = ap.parse_args()

    scene_path = os.path.abspath(args.scene_path)
    if not os.path.isdir(scene_path):
        raise SystemExit(f"Scene path not found: {scene_path}")

    # --- Read scene metadata ---
    meta_path = os.path.join(scene_path, "scene_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"scene_meta.json not found at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    width = int(meta["width"])
    height = int(meta["height"])
    num_frames = args.num_frames if args.num_frames is not None else int(meta["frame_count"])

    # --- Parse cameras_parameters.txt ---
    params_path = os.path.join(scene_path, "cameras_parameters.txt")
    cameras_parsed = parse_cameras_parameters(params_path)
    n_cams = len(cameras_parsed)
    if n_cams == 0:
        raise SystemExit(f"No cameras parsed from {params_path}")

    # Order cameras by name (matches what the trainer's natsort does later)
    cameras_parsed = sorted(cameras_parsed, key=lambda c: c["name"])
    cam_names = [c["name"] for c in cameras_parsed]

    print(f"Scene: {scene_path}")
    print(f"  resolution: {width}x{height}")
    print(f"  cameras:    {n_cams}  ({cam_names[0]}..{cam_names[-1]})")
    print(f"  frames:     {num_frames} starting at colmap_{args.starttime}")

    # --- Walk RGB and mask files per frame ---
    per_frame_rgb, per_frame_mask, _ = collect_per_frame_image_paths(scene_path, num_frames)

    # --- Build COLMAP metadata (shared across frames) ---
    # Single shared camera intrinsic (same fx,fy,cx,cy for all cams — flamsplat
    # uses the same focal length for every camera in the rig).
    # If your rig ever uses per-cam intrinsics, change this loop to write one
    # camera entry per actual camera.
    camera_id_per_cam_idx = list(range(1, n_cams + 1))
    cameras_bin_entries = []
    for ci, cam in enumerate(cameras_parsed):
        cameras_bin_entries.append(dict(
            camera_id=camera_id_per_cam_idx[ci],
            model="PINHOLE",
            width=width,
            height=height,
            # PINHOLE params: fx, fy, cx, cy. flamsplat writes fx == fy.
            params=[cam["fx"], cam["fx"], cam["cx"], cam["cy"]],
        ))

    # Image entries — one per (camera) at the starttime frame. The trainer
    # only reads images.bin at colmap_<starttime>; for later frames it
    # string-substitutes the path. We use basenames matching files we copy.
    starttime_rgb_paths = per_frame_rgb[0]
    image_basenames = []
    for ci, p in enumerate(starttime_rgb_paths):
        ext = os.path.splitext(p)[1]
        image_basenames.append(cam_names[ci] + ext)

    images_bin_entries = []
    for ci, (cam, name) in enumerate(zip(cameras_parsed, image_basenames)):
        images_bin_entries.append(dict(
            image_id=ci + 1,
            qvec=cam["qvec"],
            tvec=cam["tvec"],
            camera_id=camera_id_per_cam_idx[ci],
            name=name,
        ))

    # --- Iterate frames: copy images, sample/color points, write .bin ---
    pointsamples_dir = os.path.join(scene_path, "pointsamples")
    if not os.path.isdir(pointsamples_dir):
        raise SystemExit(
            f"pointsamples/ not found at {pointsamples_dir}. "
            f"Run the Blender 'Export Point Samples' operator first."
        )

    copy_mode = "symlink" if args.no_copy_images else "copy"

    import time
    t0 = time.monotonic()
    for fi in range(num_frames):
        idx = args.starttime + fi
        colmap_dir = os.path.join(scene_path, f"colmap_{idx}")
        images_out = os.path.join(colmap_dir, "images")
        mask_out = os.path.join(colmap_dir, "mask")
        sparse_out = os.path.join(colmap_dir, "sparse", "0")
        os.makedirs(sparse_out, exist_ok=True)

        # 1. Materialize images/
        materialize_images_dir(images_out, per_frame_rgb[fi], cam_names, mode=copy_mode)

        # 2. Materialize masks/ (optional)
        if not args.no_masks and any(p is not None for p in per_frame_mask[fi]):
            materialize_masks_dir(mask_out, per_frame_mask[fi], cam_names, mode=copy_mode)

        # 3. Load point samples for this frame
        npz_path = os.path.join(pointsamples_dir, f"{fi:05d}.npz")
        if not os.path.exists(npz_path):
            raise SystemExit(f"Missing point sample file: {npz_path}")
        npz = np.load(npz_path)
        xyz = npz["xyz"].astype(np.float64)
        normal = npz["normal"].astype(np.float64) if "normal" in npz.files else None

        if args.max_points_per_frame is not None and xyz.shape[0] > args.max_points_per_frame:
            rng = np.random.default_rng(seed=fi * 11 + 3)
            sel = rng.choice(xyz.shape[0], size=args.max_points_per_frame, replace=False)
            xyz = xyz[sel]
            if normal is not None:
                normal = normal[sel]

        # 4. Project to find each point's best camera, sample color
        cam_idx_arr, pixel_uv, valid = project_points_to_cameras(
            xyz, cameras_parsed, width, height,
            normal_world=normal if args.require_facing else None,
        )
        n_visible = int(valid.sum())
        rgb = sample_colors_from_images(cam_idx_arr, pixel_uv, valid, per_frame_rgb[fi])

        # Drop points no camera sees — they'd just be init noise
        if n_visible < xyz.shape[0]:
            xyz_kept = xyz[valid]
            rgb_kept = rgb[valid]
        else:
            xyz_kept = xyz
            rgb_kept = rgb

        # 5. Write points3D.bin (per-frame)
        write_points3d_bin(os.path.join(sparse_out, "points3D.bin"), xyz_kept, rgb_kept)

        # 6. Write cameras.bin and images.bin only at starttime folder
        if fi == 0:
            write_cameras_bin(os.path.join(sparse_out, "cameras.bin"), cameras_bin_entries)
            write_images_bin(os.path.join(sparse_out, "images.bin"), images_bin_entries)

        elapsed = time.monotonic() - t0
        eta = elapsed / (fi + 1) * (num_frames - fi - 1)
        print(
            f"[{fi + 1:>4}/{num_frames}] colmap_{idx}: "
            f"{xyz_kept.shape[0]}/{xyz.shape[0]} pts visible  "
            f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
            flush=True,
        )

    # 7. Invalidate cached merged ply so the next training run picks up the
    #    fresh per-frame points.
    cache_glob = os.path.join(scene_path, f"colmap_{args.starttime}",
                               "sparse", "0", "points3D_total*.ply")
    for stale in glob.glob(cache_glob):
        print(f"Removing stale cache: {stale}")
        os.remove(stale)

    print(f"Done. Train with: --source_path {os.path.join(scene_path, f'colmap_{args.starttime}')} "
          f"(loader=technicolor, duration={num_frames})")


if __name__ == "__main__":
    main()
