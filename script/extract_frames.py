#!/usr/bin/env python3
"""Extract frames from Cam_N.mp4 videos into cam_NNN/NNNNN.png structure."""

import os
import re
import cv2
import argparse


def extract_frames(input_dir: str, output_dir: str | None = None, fmt: str = "jpg", quality: int = 95) -> None:
    if output_dir is None:
        output_dir = input_dir

    # Find all Cam_N.mp4 files
    pattern = re.compile(r"^Cam_(\d+)\.mp4$")
    videos = []
    for f in os.listdir(input_dir):
        m = pattern.match(f)
        if m:
            videos.append((int(m.group(1)), f))

    videos.sort(key=lambda x: x[0])
    print(f"Found {len(videos)} videos in {input_dir}")

    for cam_idx, filename in videos:
        folder_name = f"cam_{cam_idx:03d}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        video_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {filename} -> {folder_name}/ ({total} frames)")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(folder_path, f"{frame_idx:05d}.{fmt}")
            params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
            cv2.imwrite(out_path, frame, params)
            frame_idx += 1

        cap.release()
        print(f"    Done: {frame_idx} frames written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from Cam_N.mp4 videos")
    parser.add_argument("input_dir", help="Directory containing Cam_N.mp4 files")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to input_dir)",
    )
    parser.add_argument("--format", default="jpg", choices=["png", "jpg"], help="Output image format (default: jpg)")
    parser.add_argument("--jpeg-quality", default=95, type=int, help="JPEG quality 1-100 (default: 95)")
    args = parser.parse_args()

    extract_frames(args.input_dir, args.output_dir, fmt=args.format, quality=args.jpeg_quality)
