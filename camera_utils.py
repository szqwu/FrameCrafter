"""
Camera utilities for Wan2.1 Novel View Synthesis.

All poses are expected in **w2c (world-to-camera) OpenCV convention**:
  - x-right, y-down, z-forward
  - This is the native COLMAP / OpenCV coordinate system.

The pipeline internally:
  1. Normalises the w2c poses so the *last* camera (target) is at the origin.
  2. Converts to c2w for raymap construction.
  3. Computes Plucker ray coordinates and pixel-unshuffles to 1/8 resolution.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core raymap functions (from DiffSynth-Studio dataset code)
# ──────────────────────────────────────────────────────────────────────────────

def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: float = 1.0,
    include_ups: bool = False,
):
    """Construct the raymap from camera intrinsics and c2w extrinsics.

    Expects OpenCV camera coordinates.

    Args:
        Ks: Camera intrinsics, shape (..., 3, 3).
        camtoworlds: Camera-to-world extrinsics, shape (..., 4, 4).
        height: Image height corresponding to intrinsics.
        width: Image width corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction.

    Returns:
        Raymap tensor, shape (..., H, W, 6) or (..., H, W, 9) if include_ups.
    """
    assert Ks.shape[-2:] == (3, 3)
    assert camtoworlds.shape[-2:] == (4, 4)

    dtype = Ks.dtype
    Ks = torch.stack(
        [Ks[..., 0, :] * downscale, Ks[..., 1, :] * downscale, Ks[..., 2, :]],
        dim=-2,
    )
    width = int(width * downscale)
    height = int(height * downscale)

    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1).to(dtype)

    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(dtype), coords)
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)


def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """Convert raymap to Plucker coordinates.

    Args:
        raymap: Shape (..., H, W, 6).

    Returns:
        Plucker coordinates, shape (..., H, W, 6).
    """
    assert raymap.shape[-1] == 6
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)


def get_plucker_rays(pose, intrinsic, height, width, no_pixel_unshuffle=False, downsample_factor=8):
    """Compute Plucker ray maps from c2w poses and intrinsics.

    Args:
        pose: (N, 4, 4) camera-to-world poses.
        intrinsic: (N, 3, 3) intrinsic matrices.
        height: Image height.
        width: Image width.
        no_pixel_unshuffle: If True, use bilinear downsampling instead of PixelUnshuffle.
        downsample_factor: Spatial downsample factor (default 8, matching VAE).

    Returns:
        Plucker ray features, shape (N, C, H/factor, W/factor).
    """
    downscale_factor = 1
    raymap = camera_to_raymap(intrinsic, pose, height=height, width=width, downscale=downscale_factor)
    plucker_ray_map = raymap_to_plucker(raymap)
    plucker_ray_map_permuted = plucker_ray_map.permute(0, 3, 1, 2)
    if not no_pixel_unshuffle:
        pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=downsample_factor)
        plucker_ray_map_permuted_unshuffled = pixel_unshuffle(plucker_ray_map_permuted)
    else:
        plucker_ray_map_permuted_unshuffled = F.interpolate(
            plucker_ray_map_permuted, scale_factor=1 / 8, mode='bilinear', align_corners=False
        )
    return plucker_ray_map_permuted_unshuffled


# ──────────────────────────────────────────────────────────────────────────────
# Pose normalisation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_w2c_make_cam_last_origin(w2c: torch.Tensor):
    """Normalise w2c poses so the *last* camera sits at the world origin.

    Steps:
      1. Convert w2c -> c2w.
      2. Align so that the last camera has identity rotation and zero translation.
      3. Scale so that the mean camera-center distance from the origin is 1.
      4. Convert back to w2c.

    Args:
        w2c: (N, 4, 4) world-to-camera extrinsics (OpenCV convention).

    Returns:
        w2c_norm: (N, 4, 4) normalised w2c.
        c2w_norm: (N, 4, 4) normalised c2w (inverse of w2c_norm).
        scale: Scalar used for the uniform scaling.
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)

    device, dtype = w2c.device, w2c.dtype
    N = w2c.shape[0]

    c2w = torch.linalg.inv(w2c)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    R0 = R[-1]
    t0 = t[-1]

    R_align = R0.transpose(0, 1)
    t_shift = t - t0
    t_rot = (R_align @ t_shift.unsqueeze(-1)).squeeze(-1)
    R_rot = R_align @ R

    dists = t_rot.norm(dim=-1)
    scale = dists.mean().clamp_min(1e-12)
    t_norm = t_rot / scale

    c2w_norm = torch.zeros_like(c2w)
    c2w_norm[:, :3, :3] = R_rot
    c2w_norm[:, :3, 3] = t_norm
    c2w_norm[:, 3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    w2c_norm = torch.linalg.inv(c2w_norm)

    return w2c_norm, c2w_norm, scale


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def prepare_raymap(w2c_poses_np, intrinsics_np, height, width):
    """Compute plucker ray features from w2c poses (OpenCV convention).

    This is the main entry point used by ``infer.py``.

    Args:
        w2c_poses_np: (N, 4, 4) numpy array -- w2c poses in OpenCV convention.
        intrinsics_np: (N, 3, 3) numpy array -- intrinsics at model resolution.
        height: Model input height.
        width: Model input width.

    Returns:
        raymap: (N, C, H/8, W/8) tensor (float32 on CPU).
    """
    w2cs = torch.from_numpy(w2c_poses_np).float()

    _, c2w_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    raymap = get_plucker_rays(
        c2w_norm,
        torch.from_numpy(intrinsics_np).float(),
        height=height,
        width=width,
    )
    return raymap
