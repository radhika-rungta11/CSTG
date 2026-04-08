#!/usr/bin/env python3
"""
Convert Dynamic_C3DGS _pp.npz checkpoint to C-friendly binary format (.4dgs.gz).

Bakes features_dc [N,6] from hash grid + MLP (time-independent, computed from
base xyz). Stores all other attributes in their existing compressed formats.
Drops hash grid, MLP weights, and rgb_dec from the binary. Keeps tfea for
runtime time-varying color.

Binary layout (all little-endian, version 3):
  Header:
    magic           4s    b"4DGS"
    version         I     3
    N               I     number of Gaussians

  Raw float16 blocks:
    xyz             [N*3] float16
    motion          [N*9] float16

  Scalar blocks × 3 (opacity, tcen, tsca):
    min_val         f
    max_val         f
    htable_len      H     Huffman table entry count
    htable entries  [htable_len * (H+B+I)]  (symbol, bitlen, code_bits)
    data_len        I     encoded byte count
    data            [data_len] uint8

  VQ blocks × 4 (scale, rotation, omega, tfea):
    num_layers      B     RVQ layer count
    cb_size         H     codebook size per layer
    dim             H     codebook vector dimension
    codebooks       [num_layers * cb_size * dim] float16
    htable_len      H
    htable entries  [htable_len * (H+B+I)]
    data_len        I
    data            [data_len] uint8

  Baked features_dc:
    has_features    B     1 if baked, 0 if GPU unavailable
    features_dc     [N*6] float16   (only if has_features == 1)

  rgb_dec (Sandwich MLP weights):
    has_rgb_dec     B     1 if present, 0 otherwise
    w1              [6*12] float16  (only if has_rgb_dec == 1)
    w2              [3*6]  float16  (only if has_rgb_dec == 1)

Usage:
    python parser.py \\
        --input path/to/point_cloud_pp.npz \\
        --output output.4dgs.gz
"""

import argparse
import gzip
import numbers
import os
import struct
import sys

import numpy as np


def to_numpy(v):
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(v, "cpu"):
        t = v.cpu().detach()
        try:
            return t.numpy(force=True)
        except TypeError:
            return t.numpy()
    return np.asarray(v)


# ── Loading ─────────────────────────────────────────────────────────────────


def load_pp_npz(path):
    """Load a Dynamic_C3DGS _pp.npz file."""
    if not os.path.isfile(path):
        candidates = [path + "_pp.npz", path + ".npz"]
        for c in candidates:
            if os.path.isfile(c):
                path = c
                break

    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)

    required = [
        "xyz",
        "motion",
        "opacity",
        "tcen",
        "tsca",
        "scale",
        "rotation",
        "omega",
        "tfea",
        "hash",
        "mlp",
        "huftable_opacity",
        "huftable_tcen",
        "huftable_tsca",
        "huftable_scale",
        "huftable_rotation",
        "huftable_omega",
        "huftable_tfea",
        "huftable_hash",
        "codebook_scale",
        "codebook_rotation",
        "codebook_omega",
        "codebook_tfea",
        "minmax_opacity",
        "minmax_tcen",
        "minmax_tsca",
        "minmax_hash",
        "rvq_info_geo",
        "rvq_info_temp",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Missing keys in {path}: {missing}\n"
            f"Available keys: {list(data.keys())}\n"
            "Is this a valid _pp.npz file from Dynamic_C3DGS?"
        )

    return data


# ── Huffman helpers ─────────────────────────────────────────────────────────


def encode_huffman_table(htable):
    """Convert a dahuffman code table dict to (symbol, bit_len, code_bits) tuples."""
    entries = []
    for key, (bit_len, code_bits) in htable.items():
        if isinstance(key, numbers.Integral):
            symbol = int(key)
        elif isinstance(key, float) and key == int(key):
            symbol = int(key)
        else:
            symbol = 0xFFFF  # _EOF sentinel
        entries.append((symbol, int(bit_len), int(code_bits)))
    return entries


def extract_vq_codebooks(state_dict):
    """Extract codebook embeddings from a ResidualVQ state dict.

    Returns list of [codebook_size, dim] float16 arrays, one per quantizer layer.
    """
    codebooks = []
    i = 0
    while True:
        key = f"layers.{i}._codebook.embed"
        if key in state_dict:
            cb = to_numpy(state_dict[key]).astype(np.float16)
            # Squeeze leading batch dim: [1, K, D] -> [K, D]
            if cb.ndim == 3 and cb.shape[0] == 1:
                cb = cb.squeeze(0)
            codebooks.append(cb)
        else:
            break
        i += 1

    if not codebooks:
        raise ValueError(
            f"No codebook embeddings found. Keys: {list(state_dict.keys())}"
        )
    return codebooks


# ── Binary writers ──────────────────────────────────────────────────────────


def write_huffman_block(buf, htable_data, encoded_data):
    """Write Huffman table + encoded byte stream."""
    htable = htable_data.item() if hasattr(htable_data, "item") else htable_data
    entries = encode_huffman_table(htable)

    buf.extend(struct.pack("<H", len(entries)))
    for sym, clen, bits in entries:
        buf.extend(struct.pack("<HBI", sym, clen, bits))

    raw = bytes(encoded_data)
    buf.extend(struct.pack("<I", len(raw)))
    buf.extend(raw)


def write_scalar_block(buf, minmax, htable_data, encoded_data):
    """Write a quantized scalar attribute: minmax + huffman block."""
    buf.extend(struct.pack("<ff", float(minmax[0]), float(minmax[1])))
    write_huffman_block(buf, htable_data, encoded_data)


def write_vq_block(buf, codebook_state_data, htable_data, encoded_data):
    """Write a VQ-compressed attribute: codebooks + huffman block."""
    state = (
        codebook_state_data.item()
        if hasattr(codebook_state_data, "item")
        else codebook_state_data
    )
    codebooks = extract_vq_codebooks(state)

    num_layers = len(codebooks)
    cb_size = codebooks[0].shape[0]
    dim = codebooks[0].shape[1]

    buf.extend(struct.pack("<BHH", num_layers, cb_size, dim))
    for cb in codebooks:
        buf.extend(cb.tobytes())

    write_huffman_block(buf, htable_data, encoded_data)


# ── Baking features_dc ──────────────────────────────────────────────────────


def contract_to_unisphere(x, aabb):
    """Contract xyz to unit sphere, matching oursfull.py:1181."""
    import torch

    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1
    mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    x = x / 4 + 0.5
    return x


def infer_hashmap_size(n_params):
    """Infer log2_hashmap_size from total hash grid parameter count."""
    import tinycudann as tcnn

    for log2_size in range(14, 24):
        enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_size,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            },
        )
        if enc.params.shape[0] == n_params:
            del enc
            return log2_size
        del enc
    raise ValueError(f"Cannot infer log2_hashmap_size for {n_params} params")


def _decode_scalar(data, htable_key, data_key, minmax_key):
    """Huffman decode + dequantize a scalar attribute from _pp.npz."""
    from dahuffman.huffmancodec import PrefixCodec

    codec = PrefixCodec(data[htable_key].item())
    values = np.array(codec.decode(data[data_key]), dtype=np.float32)
    mn, mx = float(data[minmax_key][0]), float(data[minmax_key][1])
    return (mx - mn) * values / 255.0 + mn


def try_bake_features(data):
    """Compute features_dc [N,6] from hash grid + MLP. Returns f16 ndarray or None."""
    try:
        import torch
        import tinycudann
    except ImportError as e:
        print(f"WARNING: Cannot bake features_dc — {e}")
        print("  Writing binary with has_features=0.")
        return None

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Writing binary with has_features=0.")
        return None

    device = "cuda"

    # Load xyz
    xyz = torch.from_numpy(data["xyz"].astype(np.float32)).to(device)
    N = xyz.shape[0]

    # Decode hash grid parameters
    hash_np = _decode_scalar(data, "huftable_hash", "hash", "minmax_hash")
    hash_params = torch.from_numpy(hash_np).to(device).half()

    # Infer hashmap size and create encoding
    log2_size = infer_hashmap_size(len(hash_params))
    print(f"  Hash grid: {len(hash_params)} params, log2_hashmap_size={log2_size}")

    import tinycudann as tcnn

    recolor = tcnn.Encoding(
        n_input_dims=3,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": log2_size,
            "base_resolution": 16,
            "per_level_scale": 1.447,
        },
    )
    recolor.params.data.copy_(hash_params)

    # MLP head
    mlp_params = torch.from_numpy(data["mlp"].copy()).to(device).half()
    mlp_head = tcnn.Network(
        n_input_dims=recolor.n_output_dims,
        n_output_dims=6,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    )
    mlp_head.params.data.copy_(mlp_params)

    # Compute features_dc = mlp_head(recolor(contract_to_unisphere(xyz)))
    print("  Computing features_dc via hash grid + MLP...")
    with torch.no_grad():
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
        contracted = contract_to_unisphere(xyz.clone(), aabb)
        features_dc = mlp_head(recolor(contracted)).float()  # [N, 6]

    features_dc_f16 = to_numpy(features_dc).astype(np.float16)
    print(f"  features_dc shape: {features_dc_f16.shape}")
    return features_dc_f16


# ── Main packing ────────────────────────────────────────────────────────────


def write_binary(data, output_path):
    """Pack _pp.npz data into binary format with baked features_dc."""
    xyz = data["xyz"]  # [N, 3] float16
    motion = data["motion"]  # [N, 9] float16
    N = int(xyz.shape[0])

    rvq_num_geo = int(data["rvq_info_geo"][0])
    rvq_bit_geo = int(data["rvq_info_geo"][1])
    rvq_num_temp = int(data["rvq_info_temp"][0])
    rvq_bit_temp = int(data["rvq_info_temp"][1])

    print(f"Gaussians: {N}")
    print(
        f"RVQ geo:  {rvq_num_geo} layers, {rvq_bit_geo} bits "
        f"(codebook size {int(2**rvq_bit_geo)})"
    )
    print(
        f"RVQ temp: {rvq_num_temp} layers, {rvq_bit_temp} bits "
        f"(codebook size {int(2**rvq_bit_temp)})"
    )

    # Attempt baking features_dc
    features_dc = try_bake_features(data)

    buf = bytearray()

    # ── Header ──
    buf.extend(
        struct.pack(
            "<4sII",
            b"4DGS",
            3,  # version
            N,
        )
    )

    # ── Raw float16 blocks ──
    buf.extend(xyz.tobytes())
    buf.extend(motion.tobytes())

    # ── Scalar blocks: opacity, tcen, tsca ──
    write_scalar_block(
        buf, data["minmax_opacity"], data["huftable_opacity"], data["opacity"]
    )
    write_scalar_block(buf, data["minmax_tcen"], data["huftable_tcen"], data["tcen"])
    write_scalar_block(buf, data["minmax_tsca"], data["huftable_tsca"], data["tsca"])

    # ── VQ blocks: scale, rotation, omega, tfea ──
    for attr, cb_key in [
        ("scale", "codebook_scale"),
        ("rotation", "codebook_rotation"),
        ("omega", "codebook_omega"),
        ("tfea", "codebook_tfea"),
    ]:
        write_vq_block(buf, data[cb_key], data[f"huftable_{attr}"], data[attr])

    # ── Baked features_dc ──
    if features_dc is not None:
        buf.extend(struct.pack("<B", 1))
        buf.extend(features_dc.tobytes())
        fdc_mb = features_dc.nbytes / 1024 / 1024
        print(f"features_dc: {features_dc.shape} = {fdc_mb:.2f} MB")
    else:
        buf.extend(struct.pack("<B", 0))

    # ── rgb_dec (Sandwich MLP weights for Step 2) ──
    rgb_dec = data.get("rgb_dec")
    if rgb_dec is not None:
        state = rgb_dec.item() if hasattr(rgb_dec, "item") else rgb_dec
        w1 = to_numpy(state["mlp1.weight"]).reshape(6, 12).astype(np.float16)
        w2 = to_numpy(state["mlp2.weight"]).reshape(3, 6).astype(np.float16)
        buf.extend(struct.pack("<B", 1))
        buf.extend(w1.tobytes())
        buf.extend(w2.tobytes())
        print(f"rgb_dec: w1={w1.shape}, w2={w2.shape}")
    else:
        buf.extend(struct.pack("<B", 0))

    # ── Write output ──
    total_mb = len(buf) / 1024 / 1024
    print(f"Uncompressed: {total_mb:.2f} MB")

    with gzip.open(output_path, "wb", compresslevel=9) as f:
        f.write(buf)

    gz_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Compressed:   {gz_mb:.2f} MB")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Dynamic_C3DGS _pp.npz to .4dgs.gz binary "
        "(with baked features_dc)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Model log directory (e.g. log/cube_v3) or direct path to _pp.npz",
    )
    parser.add_argument("--output", default=None, help="Output .4dgs.gz file (optional)")
    args = parser.parse_args()

    input_path = args.input

    # If input is a directory, find the latest iteration's point_cloud_pp.npz
    if os.path.isdir(input_path):
        pc_dir = os.path.join(input_path, "point_cloud")
        if not os.path.isdir(pc_dir):
            raise FileNotFoundError(f"No point_cloud/ folder found in {input_path}")
        iterations = sorted(
            [d for d in os.listdir(pc_dir) if d.startswith("iteration_")],
            key=lambda x: int(x.split("_")[1])
        )
        if not iterations:
            raise FileNotFoundError(f"No iteration folders found in {pc_dir}")
        latest = iterations[-1]
        input_path = os.path.join(pc_dir, latest, "point_cloud_pp.npz")
        print(f"Auto-detected: {input_path}")

        if args.output is None:
            out_dir = os.path.join(args.input, "output")
            os.makedirs(out_dir, exist_ok=True)
            name = os.path.basename(os.path.normpath(args.input))
            args.output = os.path.join(out_dir, f"{name}.4dgs.gz")

    if args.output is None:
        raise ValueError("--output is required when --input is a file path")

    data = load_pp_npz(input_path)
    write_binary(data, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
