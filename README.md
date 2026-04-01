# **Novel View Synthesis as Video Completion**

## Overview
We tackle the problem of sparse novel view synthesis (NVS) using video diffusion models: given *K* (≈ 5) multi-view images of a scene and their camera poses, we predict the view from a target camera pose. Many prior approaches leverage generative image priors encoded via diffusion models. However, models trained on single images lack multi-view knowledge. We instead argue that video models already contain implicit multi-view knowledge and so should be easier to adapt for NVS. Our key insight is to formulate sparse NVS as a low frame-rate video completion task. However, one challenge is that sparse NVS is defined over an unordered set of inputs, often too sparse to admit a meaningful order, so the models should be *invariant* to permutations of that input set. To this end, we present **FrameCrafter**, which adapts video models (naturally trained with coherent frame orderings) to permutation-invariant NVS through several architectural modifications, including per-frame latent encodings and removal of temporal positional embeddings. Our results suggest that video models can be easily trained to "forget" about time with minimal supervision, producing competitive performance on sparse-view NVS benchmarks.

## Environment Setup

```bash
conda create -n framecrafter python=3.11 -y
conda activate framecrafter
pip install -e .
```

## Backbone Weights

The model requires pre-trained Wan2.1 backbone weights (DiT 14B, T5 text
encoder, VAE, CLIP image encoder). Pass the directory via `base_model_dir`:

```python
model = FrameCrafter(
    "ckpt/framecrafter.safetensors",
    base_model_dir="/path/to/backbone/weights",
)
```

If `base_model_dir` is `None`, weights are downloaded automatically from
ModelScope/HuggingFace on first run.

## Model Weights

Download the FrameCrafter checkpoint:

```bash
bash prepare/download_ckpt.sh
```

This places the checkpoint at `ckpt/framecrafter.safetensors` (the default path used by the scripts).

## Quick Start

### Python API

```python
from model import FrameCrafter
import numpy as np
from PIL import Image

# Load model
model = FrameCrafter(
    "ckpt/framecrafter.safetensors",
    base_model_dir="/path/to/backbone/weights",
)

# Prepare inputs -- M context images, M+N poses (M context + N target)
M = 6  # number of context views (flexible)
images = [Image.open(f"ctx_{i}.png") for i in range(M)]
data = np.load("scene.npz")
w2c_poses = data["w2c_poses"]    # (M+N, 4, 4) -- OpenCV w2c convention
intrinsics = data["intrinsics"]  # (M+N, 3, 3) -- at original image resolution

# Generate with automatic resize + center-crop
video = model.generate(
    images=images,
    w2c_poses=w2c_poses,
    intrinsics=intrinsics,
    height=480, width=832,
    resize_mode="crop",
)

# video[:M] are context frames, video[M:] are generated novel views
for i, novel_view in enumerate(video[M:]):
    novel_view.save(f"novel_view_{i}.png")
```

### CLI

```bash
python infer.py \
    --base_model_dir /path/to/backbone/weights \
    --images ctx_0.png ctx_1.png ctx_2.png ctx_3.png ctx_4.png ctx_5.png \
    --poses_npz scene.npz \
    --height 480 --width 832 \
    --resize_mode crop \
    --output_dir results/
```

### Demo

Run on bundled example scenes. The three examples showcase different M-to-N
configurations: 6-to-1, 6-to-2, and 3-to-1.

```bash
# Run first example scene
python demo.py --base_model_dir /path/to/backbone/weights

# Run all 3 example scenes
python demo.py --base_model_dir /path/to/backbone/weights --all
```

## Input Format

### Poses (.npz)

The `.npz` file should contain:

| Key | Shape | Description |
|-----|-------|-------------|
| `w2c_poses` | (M+N, 4, 4) | World-to-camera extrinsics in OpenCV convention |
| `intrinsics` | (M+N, 3, 3) | Camera intrinsic matrices at original image resolution |

The first M entries correspond to the M context images (matching the
order of `--images`), and the remaining N entries are target camera
positions for novel view generation. Both M and N are flexible.

### Camera Conventions

All camera poses are **w2c (world-to-camera)** in **OpenCV convention**:
- x-right, y-down, z-forward

### Image Preprocessing

The model operates at a fixed resolution. Three modes are available:

1. **`resize_mode="crop"`** (recommended): Resize preserving aspect ratio,
   then center-crop to model resolution. Intrinsics are adjusted automatically.

2. **`resize_mode="stretch"`**: Plain resize ignoring aspect ratio.
   Intrinsics are adjusted automatically.

3. **`resize_mode=None`** (default): No preprocessing. Provide images and
   intrinsics already at model resolution.

Intrinsics are automatically scaled based on the input image dimensions.

## Tips

- **VRAM**: Set `--vram_limit` (in GB) to fit different GPUs. The model offloads weights to CPU when VRAM is limited. For example, use `--vram_limit 22` for a 24 GB card.
- **Speed**: Reduce `--num_inference_steps` to speed up generation (default 50; 10–20 still gives reasonable quality).

## File Structure

```
FrameCrafter/
├── model.py              # FrameCrafter class: model loading + generation
├── infer.py              # CLI for inference
├── demo.py               # Demo script for bundled examples
├── camera_utils.py       # Plucker ray computation & pose normalisation
├── assets/               # Visual examples
├── diffsynth/            # Core diffusion library
├── examples/             # Bundled demo scenes (inputs + poses)
├── prepare/              # Scripts for downloading checkpoints
├── pyproject.toml        # Package config (pip install -e .)
└── LICENSE               # Apache 2.0
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
