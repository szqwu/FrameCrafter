"""
FrameCrafter demo on bundled example scenes.

Usage:
    python demo.py --checkpoint_path checkpoint.safetensors

    # Specific scene:
    python demo.py --checkpoint_path checkpoint.safetensors \\
                   --scene example1

    # All scenes:
    python demo.py --checkpoint_path checkpoint.safetensors --all
"""

import os
import argparse

import numpy as np
from PIL import Image

from model import FrameCrafter


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def list_scenes():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return sorted(
        d for d in os.listdir(EXAMPLES_DIR)
        if os.path.isdir(os.path.join(EXAMPLES_DIR, d))
    )


def run_scene(model, scene_name, output_root, height, width):
    scene_dir = os.path.join(EXAMPLES_DIR, scene_name)
    data = np.load(os.path.join(scene_dir, "scene_data.npz"), allow_pickle=True)

    w2c_poses = data["w2c_poses"]
    intrinsics = data["intrinsics"]
    num_context = int(data["num_context"])
    num_targets = w2c_poses.shape[0] - num_context

    inputs_dir = os.path.join(scene_dir, "inputs")
    images = []
    for i in range(num_context):
        images.append(Image.open(os.path.join(inputs_dir, f"{i}.png")).convert("RGB"))

    print(f"  {num_context}-to-{num_targets} generation at {height}x{width} ...")

    video = model.generate(
        images=images,
        w2c_poses=w2c_poses,
        intrinsics=intrinsics,
        height=height,
        width=width,
        resize_mode="crop",
    )

    out_dir = os.path.join(output_root, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    for t_idx in range(num_targets):
        pred = video[num_context + t_idx]
        pred_path = os.path.join(out_dir, f"{t_idx}.png")
        pred.save(pred_path)
        print(f"  Saved: {pred_path}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="FrameCrafter Demo")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/framecrafter.safetensors",
                        help="Path to checkpoint")
    parser.add_argument("--scene", type=str, default=None,
                        help="Specific scene name to run (default: first scene)")
    parser.add_argument("--all", action="store_true",
                        help="Run all example scenes")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Output directory")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--vram_limit", type=float, default=48)
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Path to pre-downloaded backbone weights")
    args = parser.parse_args()

    scenes = list_scenes()
    if not scenes:
        print("No example scenes found in examples/ directory.")
        return

    print(f"Available example scenes: {scenes}\n")

    if args.all:
        run_scenes = scenes
    elif args.scene:
        assert args.scene in scenes, f"Scene '{args.scene}' not found. Available: {scenes}"
        run_scenes = [args.scene]
    else:
        run_scenes = [scenes[0]]

    print(f"Loading FrameCrafter model from {args.checkpoint_path} ...")
    model = FrameCrafter(
        checkpoint_path=args.checkpoint_path,
        vram_limit=args.vram_limit,
        base_model_dir=args.base_model_dir,
    )

    for i, scene in enumerate(run_scenes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(run_scenes)}] Scene: {scene}")
        print(f"{'='*60}")
        run_scene(model, scene, args.output_dir, args.height, args.width)

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
