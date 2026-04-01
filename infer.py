"""
FrameCrafter -- Novel View Synthesis Inference CLI

Supports flexible M-to-N generation: provide M context images and a
poses file with M + N camera poses.  The first M poses correspond to
the context images; the remaining N poses define target viewpoints.

Usage:
    # 6-to-1 (6 context, 1 target)
    python infer.py --checkpoint_path ckpt.safetensors \\
                    --images ctx0.png ctx1.png ctx2.png ctx3.png ctx4.png ctx5.png \\
                    --poses_npz scene_7poses.npz --output_dir results/

    # 4-to-3 (4 context, 3 targets)
    python infer.py --checkpoint_path ckpt.safetensors \\
                    --images ctx0.png ctx1.png ctx2.png ctx3.png \\
                    --poses_npz scene_7poses.npz --output_dir results/

The .npz file should contain:
    - "w2c_poses":  (M+N, 4, 4) world-to-camera poses in OpenCV convention
    - "intrinsics": (M+N, 3, 3) camera intrinsic matrices
"""

import os
import argparse

import numpy as np
from PIL import Image

from model import FrameCrafter


def main():
    parser = argparse.ArgumentParser(
        description="FrameCrafter: Novel View Synthesis Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 6-to-1 generation
  python infer.py --checkpoint_path ckpt.safetensors \\
                  --images ctx0.png ctx1.png ctx2.png ctx3.png ctx4.png ctx5.png \\
                  --poses_npz scene.npz --output_dir results/

  # 4-to-3 generation with resize+crop
  python infer.py --checkpoint_path ckpt.safetensors \\
                  --images ctx0.png ctx1.png ctx2.png ctx3.png \\
                  --poses_npz scene.npz --height 480 --width 832 \\
                  --resize_mode crop --output_dir results/
""",
    )
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/framecrafter.safetensors",
                        help="Path to .safetensors checkpoint (LoRA or full).")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Paths to M context images (any number).")
    parser.add_argument("--poses_npz", type=str, required=True,
                        help=".npz with 'w2c_poses' (M+N,4,4) and 'intrinsics' (M+N,3,3). "
                             "First M entries match --images, remaining N are targets.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save generated frames.")
    parser.add_argument("--height", type=int, default=480,
                        help="Model input height.")
    parser.add_argument("--width", type=int, default=832,
                        help="Model input width.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--vram_limit", type=float, default=48,
                        help="GPU VRAM budget in GB for offloading.")
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Path to backbone weights. None = auto-download.")
    parser.add_argument("--resize_mode", type=str, default=None,
                        choices=["crop", "stretch"],
                        help="Image preprocessing mode. None = assume model resolution.")

    args = parser.parse_args()

    context_images = [Image.open(p).convert("RGB") for p in args.images]
    data = np.load(args.poses_npz)

    print(f"Loading FrameCrafter model...")
    model = FrameCrafter(
        checkpoint_path=args.checkpoint_path,
        vram_limit=args.vram_limit,
        base_model_dir=args.base_model_dir,
    )

    print(f"Generating novel views...")
    video = model.generate(
        images=context_images,
        w2c_poses=data["w2c_poses"],
        intrinsics=data["intrinsics"],
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        resize_mode=args.resize_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    num_ctx = len(context_images)
    for i, frame in enumerate(video):
        label = "context" if i < num_ctx else "generated"
        path = os.path.join(args.output_dir, f"{label}_{i:04d}.png")
        frame.save(path)
        print(f"Saved {path}")

    print(f"\nDone! {len(video) - num_ctx} novel view(s) generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
