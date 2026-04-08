
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

## Scripts Reference

| Script | Description |
|--------|-------------|
| `script/pre_n3d.py` | Full Neural 3D preprocessing: frame extraction, COLMAP prep, and COLMAP execution |
| `script/pre_technicolor.py` | Technicolor dataset preprocessing |
| `script/extract_frames.py` | Standalone frame extraction from `Cam_N.mp4` videos |
| `script/convert_to_jpeg.py` | Batch-convert PNG images to JPEG in `cam_*/` folders |
| `script/parser.py` | Convert `_pp.npz` checkpoint to `.4dgs.gz` binary format |
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
