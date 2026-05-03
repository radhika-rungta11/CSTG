
# FLAMSPLAT
# Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields

### [[Project Page](https://maincold2.github.io/c3dgs/)] [[Extended Paper](https://arxiv.org/abs/2408.03822)]

This is an extended version of [C3DGS (CVPR24)](https://github.com/maincold2/Compact-3DGS) for dynamic scenes based on [STG@d9833e6](https://github.com/oppo-us-research/SpacetimeGaussians/tree/d9833e6c8406e7f8f6b0a437762d6d8e0758defb).

## Setup

### Installation
```shell
git clone git@github.com:nevillejs/cstg.git
cd Dynamic_C3DGS
bash script/setup.sh
```

### Dataset

 - [Neural 3D](https://github.com/facebookresearch/Neural_3D_Video.git)
 - [Technicolor](https://www.interdigital.com/data_sets/light-field-dataset)

#### Neural 3D (from videos)
```bash
conda activate colmapenv
python script/pre_n3d.py --videopath <location>/<scene> --startframe 0 --endframe 50 --format jpg
```

This extracts frames as JPEG from `cam*.mp4` videos, prepares COLMAP input, and runs COLMAP per-frame. The `--format` flag controls the output image format (`jpg` or `png`, default: `jpg`). Use `--jpeg-quality` to set JPEG quality (default: 95).

If your data is already pre-extracted as image folders (no .mp4 files), the script will skip extraction and use the existing `cam*/` frames directly.

#### Neural 3D (from pre-extracted PNG frames)
If you have an existing dataset with PNG frames and want to convert to JPEG for faster I/O:
```bash
# Convert existing PNGs to JPEG
python script/convert_to_jpeg.py <location>/<scene> --quality 95

# Then run COLMAP preprocessing
python script/pre_n3d.py --videopath <location>/<scene> --startframe 0 --endframe 50 --format jpg
```

#### Extracting frames from Cam_N.mp4 videos (standalone)
```bash
python script/extract_frames.py <input_dir> --format jpg --jpeg-quality 95
```
Extracts `Cam_N.mp4` videos into `cam_NNN/NNNNN.jpg` folder structure. Supports `--format png` for lossless output.

#### Synthetic scenes (Blender → Technicolor)

For synthetic scenes rendered with `script/flamsplat.py` (Blender add-on), prefer the Technicolor format over N3D — it carries per-camera intrinsics + principal-point offsets and skips the LLFF/poses_bounds round-trip, which fits 360° spherical camera shells better than the forward-facing LLFF assumption.

1. In Blender, run the Flamsplat panel: pick a target object, generate the camera shell, set "Export Format" to **Technicolor**, then *Render All* and *Export Technicolor*. This produces:
    - `cam_NNN/rgb/FFFFF.png` and `cam_NNN/depth/FFFFF.exr` per camera
    - `cameras_parameters.txt` (per-camera fx/cx/cy + qw qx qy qz tx ty tz, COLMAP convention)
    - `scene_meta.json` (resolution + frame count, used by the preprocessor)
2. Run the Technicolor preprocessor in flamsplat mode:
    ```bash
    python script/pre_technicolor.py --videopath <scene> --format flamsplat
    ```
    Width/height/frame-count come from `scene_meta.json`; override with `--width`, `--height`, `--num_frames` if needed. Depth maps are forwarded to `colmap_<t>/depth/camNNN.exr` so `script/depth_filter.py` keeps working.
3. Train using a Technicolor config. Copy `configs/techni_custom/template.json` to `configs/techni_custom/<scene>.json`, set `duration` to your frame count, and pick a `holdout_cam`:
    - `"holdout_cam": "stride"` → hold out every 8th camera (good for dense 360 rigs)
    - `"holdout_cam": "cam060"` → hold out a specific camera by exact name
    ```bash
    python train.py --eval --configpath configs/techni_custom/<scene>.json \
        --model_path log/<scene> --source_path <scene>/colmap_0 --comp --store_npz
    ```

## Running

### Training
```bash
conda activate dynamic_c3dgs
python train.py --quiet --eval --configpath configs/n3d_ours/<scene>.json --model_path <path to save model> --source_path <location>/<scene>/colmap_0 --comp --store_npz
```

For example:
```bash
# Neural 3D - cook_spinach
python train.py --quiet --eval --configpath configs/n3d_ours/cook_spinach.json --model_path log/ours_cook_spinach --source_path <location>/cook_spinach/colmap_0 --comp --store_npz

# Generic
python train.py --eval --configpath configs/n3d_ours/<config>.json --model_path log/<config> --source_path <location>/<config>/colmap_0 --comp --store_npz
```

The `--comp` flag applies post-processing for compression and `--store_npz` saves the compressed model in `.npz` format.

#### Checkpointing

Save checkpoints during training and resume from them:
```bash
# Save checkpoints at specific iterations
python train.py --quiet --eval --configpath configs/n3d_ours/cook_spinach.json --model_path log/ours_cook_spinach --source_path <location>/cook_spinach/colmap_0 --comp --store_npz --checkpoint_iterations 5000 10000 15000

# Resume from a specific checkpoint
python train.py --quiet --eval --configpath configs/n3d_ours/cook_spinach.json --model_path log/ours_cook_spinach --source_path <location>/cook_spinach/colmap_0 --comp --store_npz --start_checkpoint log/ours_cook_spinach/checkpoints/ckpt_005000.pth

# Resume from the latest checkpoint automatically
python train.py --quiet --eval --configpath configs/n3d_ours/cook_spinach.json --model_path log/ours_cook_spinach --source_path <location>/cook_spinach/colmap_0 --comp --store_npz --start_checkpoint latest
```

Checkpoints are saved to `<model_path>/checkpoints/ckpt_NNNNNN.pth` and contain the full training state (Gaussian parameters, neural network weights, optimizer states, densification accumulators), allowing exact resumption from any saved iteration.

<details>
<summary><span style="font-weight: bold;">More hyper-parameters in the config file</span></summary>

  Command line arguments can also set these.
  #### lambda_mask
  Weight of masking loss to control the number of Gaussians, 0.0005 by default
  #### mask_lr
  Learning rate of the masking parameter, 0.01 by default
  #### net_lr
  Learning rate for the neural field, 0.001 by default
  #### net_lr_step
  Step schedule for training the neural field
  #### max_hashmap
  Maximum hashmap size (log) of the neural field
  #### rvq_size_geo
  Codebook size in each R-VQ stage for geometric attributes
  #### rvq_num_geo
  The number of R-VQ stages for geometric attributes
  #### rvq_size_temp
  Codebook size in each R-VQ stage for temporal attributes
  #### rvq_num_temp
  The number of R-VQ stages for temporal attributes
  #### mask_prune_iter
  Pruning interval after densification, 1000 by default
  #### rvq_iter
  The iteration at which R-VQ is implemented
</details>
<br>

### Evaluation

```bash
# Neural 3D
python test.py --quiet --eval --skip_train --valloader colmapvalid --configpath configs/n3d_ours/<scene>.json --model_path <path to model>
# Technicolor
python test.py --quiet --eval --skip_train --valloader technicolorvalid --configpath configs/techni_ours/<scene>.json --model_path <path to model>
```

### Floater Filter (post-training, optional)

Use the GT depth maps as a geometric oracle to prune mid-air "floater" Gaussians from a trained model. Each Gaussian center is projected into every training camera; Gaussians that sit significantly in front of the GT surface, or that project into sky pixels while at scene-realistic depth, are flagged and removed. The script decodes the Huffman+RVQ compressed `_pp.npz`, subsets all per-Gaussian arrays consistently, and re-encodes — so the downstream `parser.py` works unchanged on the filtered output.

**Prerequisites**: `pre_n3d.py` must have been run with depth maps present at `<scene>/cam_*/depth/<frame>.{exr,jpg,png}`; it forwards them into `colmap_<t>/depth/<cam_name>.{exr,jpg,png}`, where the filter reads them. EXR is preferred for full float-precision depth (e.g. Blender's Z-pass).

```bash
# Filter the latest iteration's _pp.npz; writes to point_cloud/iteration_<N>_filtered/
python script/depth_filter.py --model log/ours_cook_spinach --source <location>/cook_spinach/colmap_0

# Then re-package the filtered model
python script/parser.py \
    --input log/ours_cook_spinach/point_cloud/iteration_25000_filtered/point_cloud_pp.npz \
    --output log/ours_cook_spinach_filtered.4dgs.gz
```

Always start with `--dry-run` to see the floater-score distribution and choose thresholds before writing.

**Single-frame (static scene, default)**:
```bash
python script/depth_filter.py --model log/<run> --source <scene>/colmap_0 --dry-run
```

**Multi-frame (dynamic scene)** — evaluates the per-Gaussian motion polynomial at each frame, weights contributions by trbf temporal activity, and reads `<scene>/colmap_<t>/depth/` for each `t`. The number of training frames must match `--duration`:
```bash
python script/depth_filter.py --model log/<run> --source <scene>/colmap_0 --duration 50 --dry-run
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Path to model dir, e.g. `log/ours_cook_spinach`. Latest iteration in `point_cloud/` is used unless `--iteration` is given. |
| `--source` | required | Path to COLMAP data dir, e.g. `<scene>/colmap_0`. Provides camera poses + GT depth (multi-frame walks `colmap_<t>/` siblings). |
| `--threshold` | `0.5` | Floater-score cutoff. Prune if Gaussian is "in front of surface" (or in sky) in more than this fraction of views. Lower = more aggressive. |
| `--in-front` | `1.0` | Meters by which Gaussian's camera-space `z` must be less than the GT surface depth to count as in-front-of-surface in that view. Tighten (e.g. `0.05`) to catch sub-meter floaters near foliage edges. |
| `--min-views` | `3` | Auto-prune Gaussians visible in fewer than this many cameras (likely outliers). |
| `--opacity-min` | off | If set (e.g. `0.05`), additionally drop Gaussians whose `sigmoid(_opacity)` is below this. Targets translucent ghosts. |
| `--iteration` | latest | Specific iteration folder to filter, e.g. `25000`. Defaults to the highest `iteration_<N>` present. |
| `--dry-run` | off | Compute and report stats only; don't write the filtered npz. |
| `--duration` | `1` | Number of trained frames. >1 enables multi-frame mode (motion polynomial + per-frame depth). |
| `--uniform-time` | off | Multi-frame only: weight every (Gaussian, frame) pair equally instead of by trbf temporal activity. |
| `--min-weight` | `0.05` | Multi-frame only: skip frames where a Gaussian's trbf activity weight is below this. |

#### Tuning notes

- **Default `--threshold 0.5 --in-front 1.0`** is conservative — drops mostly outliers and obvious mid-air ghosts.
- **`--threshold 0.3 --in-front 0.05`** is aggressive — recommended for tree/foliage scenes where floaters can sit close to legitimate geometry.
- The filter catches mid-air floaters (geometric ghosts) and sky-region floaters (Gaussians projecting into sky pixels at scene-realistic depth). It does **not** catch wrong-color Gaussians sitting on real surfaces — those need a different filter (per-view RGB consistency).
- A `filter_meta.npz` is saved next to the filtered `_pp.npz` containing `floater_score`, `in_view`, `sky_count`, and `keep_mask` arrays for downstream inspection.

### Exporting to binary format

Convert a trained `_pp.npz` checkpoint to a C-friendly binary format (`.4dgs.gz`) with baked `features_dc`:

```bash
# From a model directory (auto-detects latest iteration)
python script/parser.py --input log/ours_cook_spinach

# From a specific npz file
python script/parser.py --input log/ours_cook_spinach/point_cloud/iteration_25000/point_cloud_pp.npz --output output.4dgs.gz
```

The binary format bakes the hash grid + MLP into precomputed `features_dc`, stores all attributes in their compressed formats (Huffman + RVQ), and includes the Sandwich MLP weights for runtime color decoding.

## Image Format Support

The pipeline supports both PNG and JPEG images. JPEG is recommended for training as it provides significant disk and I/O benefits with negligible quality impact:

| | PNG | JPEG (Q95) |
|---|---|---|
| Typical size (2704x2028) | ~9.5 MB | ~1.1 MB |
| Disk usage (128 cams x 17 frames) | ~55 GB | ~3.8 GB |
| COLMAP compatibility | Yes | Yes |
| Alpha channel | Yes | No |

The training loader (`dataset_readers.py`) includes a format-agnostic fallback: if COLMAP metadata references `.png` but only `.jpg` exists on disk (or vice versa), the loader will find the correct file automatically.

## Hyperparameter Tuning

Automated Bayesian hyperparameter optimization using Optuna. The tuner searches over 18 parameters (learning rates, densification, model capacity, etc.) and uses PSNR as the objective.

### Setup
```bash
pip install optuna optuna-dashboard
```

### Single-objective (maximize PSNR)
```bash
python script/optuna_tuner.py \
  --source_path data/horse_ol/colmap_0 \
  --base_config configs/custom_default.json \
  --n_trials 20 \
  --study_name horse_tuning \
  --output_dir optuna_runs
```

Bad trials are pruned early (killed at iter 5000+ if PSNR is below median), saving GPU time. First 3 trials always run to completion to build a baseline.

### Multi-objective (maximize PSNR + minimize gaussian count)
```bash
python script/optuna_tuner.py \
  --source_path data/horse_ol/colmap_0 \
  --base_config configs/custom_default.json \
  --n_trials 20 \
  --study_name horse_pareto \
  --output_dir optuna_runs \
  --multi_objective
```

Finds the Pareto front of configs that trade off quality vs model size.

### Monitoring

Training outputs `metrics.json` every 100 iterations with PSNR, SSIM, loss, and gaussian count. View live results with the Optuna dashboard:
```bash
optuna-dashboard sqlite:///<output_dir>/<study_name>.db --port 8081 --host 0.0.0.0
```

Access via SSH tunnel: `ssh -L 8081:localhost:8081 <your-ssh-command>`

Each trial's training log is at `<output_dir>/<study_name>/trial_NNN/stdout.log`.

### Custom datasets

Use `configs/custom_default.json` as a starting point. Set `duration` to your frame count and `resolution` to 1 (full-res) or 2 (half-res). The tuner explores the rest:

| Parameter | Range | Description |
|-----------|-------|-------------|
| scaling_lr | [0.0003, 0.005] | Scale learning rate |
| opacity_lr | [0.01, 0.1] | Opacity learning rate |
| mask_lr | [0.003, 0.02] | Mask learning rate |
| densify_grad_threshold | [0.00003, 0.0005] | Gradient threshold for densification |
| gnumlimit | [300K, 3M] | Max gaussian count |
| densify_until_iter | [10K, 30K] | When to stop densifying |
| lambda_mask | [0.0002, 0.005] | Mask loss weight |
| lambda_dssim | [0.1, 0.4] | SSIM weight in loss |
| max_hashmap | [14, 18] | Hash grid size (log2) |
| rvq_size_geo/temp | [256, 512, 1024] | RVQ codebook size |
| rvq_num_geo/temp | [3, 5] | RVQ layers |
| iterations | [20K, 45K] | Training length |
| emsstart | [6K, 20K] | Error-guided sampling start |
| desicnt | [6, 18] | Densification count |
| mask_prune_iter | [500, 2000] | Pruning interval |

## Scripts Reference

| Script | Description |
|--------|-------------|
| `script/pre_n3d.py` | Full Neural 3D preprocessing: frame extraction, COLMAP prep, and COLMAP execution |
| `script/pre_technicolor.py` | Technicolor dataset preprocessing |
| `script/extract_frames.py` | Standalone frame extraction from `Cam_N.mp4` videos |
| `script/convert_to_jpeg.py` | Batch-convert PNG images to JPEG in `cam_*/` folders |
| `script/parser.py` | Convert `_pp.npz` checkpoint to `.4dgs.gz` binary format |
| `script/depth_filter.py` | Post-training floater filter using GT depth maps as a geometric oracle (single + multi-frame) |
| `script/optuna_tuner.py` | Optuna hyperparameter tuner (single or multi-objective) |
| `script/post.py` | Post-processing utilities (video generation, metrics) |
| `script/setup.sh` | Environment setup script |

## Real-time Viewer
Without --comp and --store_npz options, our code saves the models in the original STG format, which can be used for the STG's viewer.

## Acknowledgements
A great thanks to the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [STG](https://github.com/oppo-us-research/SpacetimeGaussians) for their amazing work. For more details, please check out their repos.

## BibTeX
```
@article{Lee_2024_C3DGS,
  title={Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields},
  author={Lee, You Chan and Rho, Daniel and Sun, Xiangyu and Ko, Jong Hwan and Park, Eunbyung},
  journal={arXiv preprint arXiv:2408.03822},
  year={2024}
}
@InProceedings{Lee_2024_CVPR,
  author    = {Lee, You Chan and Rho, Daniel and Sun, Xiangyu and Ko, Jong Hwan and Park, Eunbyung},
  title     = {Compact 3D Gaussian Representation for Radiance Field},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  pages     = {21719-21728}
}
```
