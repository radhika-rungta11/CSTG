#!/usr/bin/env python3
"""
Post-training floater filter using GT depth maps as a geometric oracle.

For each Gaussian center, projects it into every training camera and checks
whether it sits significantly in front of the GT surface in that view.
Gaussians flagged as in-front-of-surface in most views are floaters and pruned.

This separates concerns cleanly:
  - Training optimizes for fidelity (max info retention)
  - This script uses the verified GT depth pipeline as a *geometric oracle*
    to remove floaters without touching legitimate-but-low-opacity Gaussians.

Usage:
    python script/depth_filter.py \\
        --model log/tree_v3_minimal \\
        --source data/tree/colmap_0
    # then:
    python script/parser.py \\
        --input log/tree_v3_minimal/point_cloud/iteration_25000_filtered/point_cloud_pp.npz \\
        --output log/tree_v3_minimal_filtered.4dgs.gz
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2  # noqa: E402

from dahuffman.huffmancodec import PrefixCodec  # noqa: E402


def _load_colmap_loader():
    """Load colmap_loader.py directly to avoid pulling in the full `scene`
    package (which triggers training-time imports)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    path = os.path.join(repo_root, "thirdparty", "gaussian_splatting",
                        "scene", "colmap_loader.py")
    spec = importlib.util.spec_from_file_location("colmap_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cl = _load_colmap_loader()
qvec2rotmat = _cl.qvec2rotmat
read_extrinsics_binary = _cl.read_extrinsics_binary
read_intrinsics_binary = _cl.read_intrinsics_binary


# ── Camera + depth loading ──────────────────────────────────────────────────


def load_cameras(colmap_path):
    """Return dict of cam_name -> {R, T (world-to-cam), fx, fy, cx, cy, W, H, image_name}."""
    extr = read_extrinsics_binary(os.path.join(colmap_path, "sparse/0/images.bin"))
    intr = read_intrinsics_binary(os.path.join(colmap_path, "sparse/0/cameras.bin"))
    cams = {}
    for k, e in extr.items():
        i = intr[e.camera_id]
        # COLMAP stores world->camera (qvec, tvec); we use directly
        R = qvec2rotmat(e.qvec)
        T = np.array(e.tvec, dtype=np.float64)
        if i.model == "PINHOLE":
            fx, fy, cx, cy = i.params
        elif i.model == "SIMPLE_PINHOLE":
            fx, cx, cy = i.params
            fy = fx
        else:
            raise ValueError(f"Unsupported camera model: {i.model}")
        cams[os.path.splitext(e.name)[0]] = dict(
            R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy,
            W=int(i.width), H=int(i.height), image_name=e.name,
        )
    return cams


def _find_scene_meta(source_path):
    """Look for scene_meta.json in the source dir or its parent.

    flamsplat writes scene_meta.json at the dataset root. depth_filter is
    typically invoked with --source <dataset>/colmap_0, so we check both.
    """
    parent = os.path.dirname(os.path.abspath(source_path).rstrip("/"))
    for c in (
        os.path.join(source_path, "scene_meta.json"),
        os.path.join(parent, "scene_meta.json"),
    ):
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f)
    return None


def load_depth(colmap_path, image_name, depth_near=0.0, depth_far=None):
    base = os.path.splitext(image_name)[0]
    for ext in (".exr", ".png", ".jpg"):
        p = os.path.join(colmap_path, "depth", base + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                return None
            if img.ndim == 3:
                img = img[:, :, 0]
            # PNG-16 ticks → meters: flamsplat encodes
            #   pixel = ((z - near) / (far - near)) * 65535
            # so the decode is z = near + (pixel/65535) * (far - near).
            # EXR / 32-bit float images are already in meters.
            if img.dtype == np.uint16:
                if depth_far is None:
                    raise RuntimeError(
                        f"{p}: 16-bit PNG depth needs depth_far. Pass --depth-far "
                        f"or place scene_meta.json with 'depth_far' near --source."
                    )
                near_f = float(depth_near)
                span = float(depth_far) - near_f
                return near_f + (img.astype(np.float32) / 65535.0) * span
            return img.astype(np.float32)
    return None


# ── Geometric scoring ───────────────────────────────────────────────────────


def _evaluate_position(xyz, motion, t_offset):
    """Evaluate Gaussian centers at a per-Gaussian time offset.

    pos(g) = xyz[g] + motion[g, 0:3]*t + motion[g, 3:6]*t² + motion[g, 6:9]*t³
    """
    t = t_offset[:, None]  # [N, 1]
    return (xyz
            + motion[:, 0:3] * t
            + motion[:, 3:6] * (t * t)
            + motion[:, 6:9] * (t * t * t))


def _project_one(pos, cam, depth, in_front_threshold,
                 valid_min, valid_max, scene_z_max):
    """Run a single (Gaussian-batch, camera, depth) projection pass.
    Returns boolean masks: valid_geom, in_sky, is_floater_geom, z."""
    N = pos.shape[0]
    Pc = (cam["R"] @ pos.T).T + cam["T"]
    z = Pc[:, 2]
    in_front_of_cam = z > 0.1
    z_safe = np.where(z > 0.1, z, 1.0)
    u = cam["fx"] * Pc[:, 0] / z_safe + cam["cx"]
    v = cam["fy"] * Pc[:, 1] / z_safe + cam["cy"]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < cam["W"]) & (vi >= 0) & (vi < cam["H"])
    valid = in_front_of_cam & in_bounds

    gt_d = np.zeros(N, dtype=np.float32)
    if valid.any():
        gt_d[valid] = depth[vi[valid], ui[valid]]

    valid_geom = valid & (gt_d > valid_min) & (gt_d < valid_max)
    in_sky = valid & (gt_d >= valid_max) & (z > 0.5) & (z < scene_z_max)
    is_floater_geom = valid_geom & (z < gt_d - in_front_threshold)
    return valid_geom, in_sky, is_floater_geom


def compute_floater_scores(xyz, cams, source_path,
                           motion=None, tcen=None, tsca=None, duration=1,
                           in_front_threshold=1.0,
                           valid_min=0.05, valid_max=100.0,
                           scene_z_max=100.0,
                           min_weight=0.05, time_weighted=True,
                           depth_near=0.0, depth_far=None):
    """For each Gaussian center, score = fraction of views where the Gaussian
    is either (a) significantly in front of the GT surface, OR (b) projects
    into a sky pixel while sitting at scene-realistic depth (mid-air floater
    over sky).

    Multi-frame: if duration > 1, motion+tcen+tsca must be provided.
    Position at time t is evaluated via the polynomial motion model;
    contributions from each (Gaussian, frame, camera) tuple are weighted by
    the Gaussian's temporal activity weight at that frame (unless
    time_weighted=False).

    Static (duration=1): motion/tcen/tsca are ignored; positions are xyz.

    Returns: floater_score [N], in_view_count [N], sky_count [N]
    """
    N = xyz.shape[0]
    floater_sum = np.zeros(N, dtype=np.float64)
    in_view_sum = np.zeros(N, dtype=np.float64)
    sky_sum = np.zeros(N, dtype=np.float64)

    if duration > 1:
        # source_path points at colmap_0; parent dir holds colmap_<t> for each frame
        source_parent = os.path.dirname(os.path.abspath(source_path).rstrip("/"))
        if motion is None or tcen is None or tsca is None:
            raise ValueError("motion/tcen/tsca required when duration > 1")

    print(f"Projecting {N:,} Gaussians × {duration} frame(s) × {len(cams)} cams"
          f" {'(time-weighted)' if (duration > 1 and time_weighted) else ''}...")

    for t_idx in range(duration):
        # Per-Gaussian time offset; for static scenes this is just zero.
        if duration > 1:
            t_norm = (t_idx + 0.5) / duration  # frame midpoint
            t_offset = t_norm - tcen
            weight_t = np.exp(-(t_offset / np.maximum(np.exp(tsca), 1e-6)) ** 2)
            active_mask = weight_t > min_weight
            if not active_mask.any():
                continue
            pos_t = _evaluate_position(xyz, motion, t_offset)
            if time_weighted:
                w = weight_t.astype(np.float64)
            else:
                w = active_mask.astype(np.float64)
            frame_dir = os.path.join(source_parent, f"colmap_{t_idx}")
        else:
            pos_t = xyz
            w = np.ones(N, dtype=np.float64)
            frame_dir = source_path

        for ci, (cam_name, cam) in enumerate(cams.items()):
            depth = load_depth(frame_dir, cam["image_name"],
                               depth_near=depth_near, depth_far=depth_far)
            if depth is None:
                continue
            if depth.shape != (cam["H"], cam["W"]):
                depth = cv2.resize(depth, (cam["W"], cam["H"]),
                                   interpolation=cv2.INTER_NEAREST)

            valid_geom, in_sky, is_floater_geom = _project_one(
                pos_t, cam, depth, in_front_threshold,
                valid_min, valid_max, scene_z_max)

            floater_sum += w * (is_floater_geom | in_sky).astype(np.float64)
            in_view_sum += w * (valid_geom | in_sky).astype(np.float64)
            sky_sum += w * in_sky.astype(np.float64)

        if duration > 1:
            print(f"  processed frame {t_idx + 1}/{duration}")

    if duration == 1:
        # Match prior behavior: integer counts for single-frame stats.
        floater_count = floater_sum.astype(np.int64)
        in_view_count = in_view_sum.astype(np.int64)
        sky_count = sky_sum.astype(np.int64)
    else:
        # Multi-frame: keep float weights for percentile reporting clarity.
        floater_count = floater_sum
        in_view_count = in_view_sum
        sky_count = sky_sum

    floater_score = floater_sum / np.maximum(in_view_sum, 1e-6)
    return floater_score, in_view_count, sky_count


# ── npz filtering / re-encoding ─────────────────────────────────────────────


def _decode_scalar_pp(data, attr):
    """Decode a huffman+minmax-quantized scalar attribute from _pp.npz."""
    codec = PrefixCodec(data[f"huftable_{attr}"].item())
    symbols = np.asarray(list(codec.decode(data[attr])), dtype=np.float32)
    mn, mx = float(data[f"minmax_{attr}"][0]), float(data[f"minmax_{attr}"][1])
    return (mx - mn) * symbols / 255.0 + mn


def _decode(codec, byte_data):
    return list(codec.decode(byte_data))


def _encode(codec, symbols):
    out = codec.encode(symbols)
    if isinstance(out, (bytes, bytearray)):
        return np.frombuffer(out, dtype=np.uint8).copy()
    if isinstance(out, np.ndarray):
        return out.astype(np.uint8, copy=False)
    return np.array(list(out), dtype=np.uint8)


def filter_npz(input_path, keep_mask, output_path):
    data = np.load(input_path, allow_pickle=True)
    N_orig = int(data["xyz"].shape[0])
    N_keep = int(keep_mask.sum())
    print(f"Loading {input_path}")
    print(f"  Original Gaussians: {N_orig:,}")
    print(f"  Keep:               {N_keep:,} ({100 * N_keep / N_orig:.2f}%)")
    print(f"  Drop:               {N_orig - N_keep:,} "
          f"({100 * (N_orig - N_keep) / N_orig:.2f}%)")

    out = {k: data[k] for k in data.keys()}

    # Raw float16 per-Gaussian arrays: trivial subset
    out["xyz"] = data["xyz"][keep_mask]
    out["motion"] = data["motion"][keep_mask]

    # Scalar attrs (one symbol per Gaussian, quantized to 0..255)
    for attr in ("opacity", "tcen", "tsca"):
        codec = PrefixCodec(data[f"huftable_{attr}"].item())
        symbols = _decode(codec, data[attr])
        if len(symbols) != N_orig:
            raise RuntimeError(
                f"{attr}: decoded {len(symbols)} symbols but expected {N_orig}"
            )
        symbols_arr = np.asarray(symbols, dtype=np.int64)
        filtered = symbols_arr[keep_mask]
        out[attr] = _encode(codec, filtered.tolist())

    # RVQ-quantized attrs: detect layer count from the actual decoded symbol count
    # (rvq_info_geo/temp don't reliably map to which attribute uses which).
    for attr in ("scale", "rotation", "omega", "tfea"):
        codec = PrefixCodec(data[f"huftable_{attr}"].item())
        symbols = _decode(codec, data[attr])
        if len(symbols) % N_orig != 0:
            raise RuntimeError(
                f"{attr}: decoded {len(symbols)} symbols not divisible by "
                f"N_orig={N_orig}"
            )
        n_layers = len(symbols) // N_orig
        symbols_per_g = np.asarray(symbols, dtype=np.int64).reshape(N_orig, n_layers)
        filtered = symbols_per_g[keep_mask].flatten()
        out[attr] = _encode(codec, filtered.tolist())

    np.savez_compressed(output_path, **out)
    print(f"Saved filtered _pp.npz to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model dir, e.g. log/tree_v3_minimal")
    ap.add_argument("--source", required=True, help="COLMAP path, e.g. data/tree/colmap_0")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Floater score cutoff: prune if in-front-of-surface in >threshold of views (default 0.5)")
    ap.add_argument("--in-front", type=float, default=1.0,
                    help="Min meters Gaussian must sit in front of surface to count (default 1.0)")
    ap.add_argument("--min-views", type=int, default=3,
                    help="Skip Gaussians visible in fewer than this many cameras (default 3)")
    ap.add_argument("--iteration", type=int, default=None,
                    help="Iteration to filter (default: latest)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report stats only; don't write filtered npz")
    ap.add_argument("--opacity-min", type=float, default=None,
                    help="Drop Gaussians whose sigmoid(_opacity) is below this threshold "
                         "(e.g. 0.05 to drop translucent ghosts). Off by default.")
    ap.add_argument("--duration", type=int, default=1,
                    help="Number of training frames. >1 enables multi-frame mode: "
                         "evaluates motion polynomial per frame, loads depth from "
                         "colmap_<t>/depth/, weights contributions by trbf activity.")
    ap.add_argument("--uniform-time", action="store_true",
                    help="Multi-frame only: weight every (Gaussian, frame) pair "
                         "uniformly instead of by trbf activity (default: time-weighted).")
    ap.add_argument("--min-weight", type=float, default=0.05,
                    help="Multi-frame only: skip frames where Gaussian's trbf activity "
                         "weight is below this (default 0.05).")
    ap.add_argument("--depth-far", type=float, default=None,
                    help="Far-clip in meters used to scale 16-bit PNG depth back to "
                         "meters. Auto-discovered from scene_meta.json (next to --source "
                         "or its parent) if not set. Ignored for EXR/float depth inputs.")
    ap.add_argument("--depth-near", type=float, default=None,
                    help="Near-clip in meters paired with --depth-far for the inverse "
                         "of flamsplat's PNG-16 encoding. Auto-discovered from "
                         "scene_meta.json. Defaults to 0.0 if missing.")
    args = ap.parse_args()

    # Resolve depth_near/far for uint16 PNG depth (no-op for float/EXR inputs).
    depth_far = args.depth_far
    depth_near = args.depth_near
    if depth_far is None or depth_near is None:
        meta = _find_scene_meta(args.source)
        if meta:
            if depth_far is None and "depth_far" in meta:
                depth_far = float(meta["depth_far"])
            if depth_near is None and "depth_near" in meta:
                depth_near = float(meta["depth_near"])
            if depth_far is not None or depth_near is not None:
                print(f"Auto-detected depth_near={depth_near} depth_far={depth_far}"
                      f" from scene_meta.json")
    if depth_near is None:
        depth_near = 0.0

    # Locate _pp.npz
    pc_dir = os.path.join(args.model, "point_cloud")
    iters = sorted(
        [d for d in os.listdir(pc_dir) if d.startswith("iteration_") and d[10:].isdigit()],
        key=lambda x: int(x.split("_")[1]),
    )
    if not iters:
        raise FileNotFoundError(f"No iteration_NNN folders in {pc_dir}")
    chosen = (f"iteration_{args.iteration}" if args.iteration is not None
              else iters[-1])
    iter_dir = os.path.join(pc_dir, chosen)
    pp_path = os.path.join(iter_dir, "point_cloud_pp.npz")
    if not os.path.isfile(pp_path):
        raise FileNotFoundError(pp_path)
    print(f"Model: {pp_path}")

    # Load Gaussian centers
    pp = np.load(pp_path, allow_pickle=True)
    xyz = pp["xyz"].astype(np.float64)  # float64 for projection precision
    print(f"  Gaussians: {xyz.shape[0]:,}")
    print(f"  xyz range: x[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] "
          f"y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] "
          f"z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    # Load cameras + GT depth
    cams = load_cameras(args.source)
    print(f"  Cameras: {len(cams)}")

    # If multi-frame, decode motion + tcen + tsca and verify per-frame depth dirs
    motion = tcen = tsca = None
    if args.duration > 1:
        motion = pp["motion"].astype(np.float32).reshape(-1, 9)
        tcen = _decode_scalar_pp(pp, "tcen")
        tsca = _decode_scalar_pp(pp, "tsca")
        print(f"  Motion polynomial: 9-coeff (linear+quad+cubic in xyz)")
        print(f"  trbf center  range: [{tcen.min():.3f}, {tcen.max():.3f}]")
        print(f"  trbf scale (log) range: [{tsca.min():.3f}, {tsca.max():.3f}]"
              f"  → activity FWHM ≈ "
              f"{2 * np.sqrt(np.log(2)) * np.exp(np.median(tsca)):.3f}")
        # Sanity check colmap_<t> dirs exist
        source_parent = os.path.dirname(os.path.abspath(args.source).rstrip("/"))
        missing = [t for t in range(args.duration)
                   if not os.path.isdir(os.path.join(source_parent, f"colmap_{t}"))]
        if missing:
            print(f"  WARNING: missing colmap_<t> dirs for frames: {missing[:10]}"
                  + ("..." if len(missing) > 10 else ""))

    # Score
    floater_score, in_view, sky_count = compute_floater_scores(
        xyz, cams, args.source,
        motion=motion, tcen=tcen, tsca=tsca, duration=args.duration,
        in_front_threshold=args.in_front,
        min_weight=args.min_weight, time_weighted=not args.uniform_time,
        depth_near=depth_near, depth_far=depth_far)

    # Stats
    print("\nFloater-score distribution:")
    for q in (0.5, 0.7, 0.9, 0.95, 0.99):
        print(f"  {int(q * 100):>3}th pctile: score={np.quantile(floater_score, q):.3f}")
    print(f"  mean score: {floater_score.mean():.3f}")
    print(f"  in-view-count median: {int(np.median(in_view))}, max: {int(in_view.max())}")
    print(f"  sky-projection median: {int(np.median(sky_count))}, "
          f"max: {int(sky_count.max())} "
          f"({100 * np.mean(sky_count > 5):.1f}% of Gaussians project to sky in >5 views)")
    n_low_visibility = int((in_view < args.min_views).sum())
    if n_low_visibility:
        print(f"  Gaussians visible in <{args.min_views} cams (auto-prune): {n_low_visibility:,}")

    # Build keep mask: prune if floater_score >= threshold OR visible in too few views
    keep_mask = (floater_score < args.threshold) & (in_view >= args.min_views)

    # Optional opacity filter: drop translucent ghosts
    opacity_drop_count = 0
    if args.opacity_min is not None:
        codec = PrefixCodec(pp["huftable_opacity"].item())
        symbols = list(codec.decode(pp["opacity"]))
        symbols_arr = np.asarray(symbols, dtype=np.float32)
        mn, mx = float(pp["minmax_opacity"][0]), float(pp["minmax_opacity"][1])
        raw_opacity = (mx - mn) * symbols_arr / 255.0 + mn
        sig_opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        opacity_keep = sig_opacity >= args.opacity_min
        opacity_drop_count = int((~opacity_keep & keep_mask).sum())
        keep_mask = keep_mask & opacity_keep
        print(f"\nOpacity stats (sigmoid-decoded):")
        print(f"  min/median/max: {sig_opacity.min():.3f} / "
              f"{np.median(sig_opacity):.3f} / {sig_opacity.max():.3f}")
        print(f"  Below {args.opacity_min}: "
              f"{(sig_opacity < args.opacity_min).sum():,} Gaussians "
              f"({100*(sig_opacity < args.opacity_min).mean():.1f}%)")

    n_keep = int(keep_mask.sum())
    n_drop = len(keep_mask) - n_keep
    print(f"\nFiltering with threshold={args.threshold}, in_front={args.in_front}m, "
          f"min_views={args.min_views}"
          + (f", opacity_min={args.opacity_min}" if args.opacity_min is not None else "")
          + ":")
    print(f"  Keep: {n_keep:,} ({100 * n_keep / len(keep_mask):.2f}%)")
    print(f"  Drop: {n_drop:,} ({100 * n_drop / len(keep_mask):.2f}%)")
    if opacity_drop_count:
        print(f"    (incremental opacity drop: {opacity_drop_count:,})")

    if args.dry_run:
        print("\n[dry-run] Skipping write.")
        return

    # Save filtered _pp.npz to a sibling iteration folder
    out_iter_dir = os.path.join(pc_dir, chosen + "_filtered")
    os.makedirs(out_iter_dir, exist_ok=True)
    out_path = os.path.join(out_iter_dir, "point_cloud_pp.npz")
    filter_npz(pp_path, keep_mask, out_path)

    # Also save the keep mask + scores for inspection
    np.savez(os.path.join(out_iter_dir, "filter_meta.npz"),
             floater_score=floater_score, in_view=in_view,
             sky_count=sky_count, keep_mask=keep_mask)
    print(f"Filter metadata saved to {out_iter_dir}/filter_meta.npz")
    print("\nNext step:")
    print(f"  python script/parser.py --input {out_path} \\")
    print(f"      --output {args.model}_filtered.4dgs.gz")


if __name__ == "__main__":
    main()
