"""
FrameCrafter -- Novel View Synthesis powered by Wan2.1

This module contains the FrameCrafter class which handles model loading,
checkpoint management, and novel-view generation.

Usage:
    from model import FrameCrafter
    import numpy as np
    from PIL import Image

    model = FrameCrafter("checkpoint.safetensors")
    video = model.generate(images=[...], w2c_poses=..., intrinsics=...)
"""

import os
import gc
import glob as _glob

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.core.vram.layers import AutoWrappedModule

from camera_utils import prepare_raymap


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def resize_and_center_crop(img_np, tgt_h, tgt_w):
    """Resize preserving aspect ratio to cover (tgt_h, tgt_w), then center-crop.

    Args:
        img_np: (H, W, 3) uint8 numpy array.
        tgt_h, tgt_w: Target output size.

    Returns:
        cropped: (tgt_h, tgt_w, 3) uint8 numpy array.
        crop_params: dict with resize_w, resize_h, offset_x, offset_y.
    """
    h, w = img_np.shape[:2]
    tgt_aspect = tgt_w / tgt_h
    src_aspect = w / h

    if src_aspect > tgt_aspect:
        resize_h = tgt_h
        resize_w = int(round(w * tgt_h / h))
    else:
        resize_w = tgt_w
        resize_h = int(round(h * tgt_w / w))

    resized = cv2.resize(img_np, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    offset_x = (resize_w - tgt_w) // 2
    offset_y = (resize_h - tgt_h) // 2
    cropped = resized[offset_y : offset_y + tgt_h, offset_x : offset_x + tgt_w]

    crop_params = {
        "resize_w": resize_w, "resize_h": resize_h,
        "offset_x": offset_x, "offset_y": offset_y,
    }
    return cropped, crop_params


def resize_stretch(img_np, tgt_h, tgt_w):
    """Plain resize to (tgt_h, tgt_w), ignoring aspect ratio.

    Args:
        img_np: (H, W, 3) uint8 numpy array.
        tgt_h, tgt_w: Target output size.

    Returns:
        resized: (tgt_h, tgt_w, 3) uint8 numpy array.
        crop_params: dict (offset_x/y are 0; resize_w/h equal tgt_w/h).
    """
    resized = cv2.resize(img_np, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
    crop_params = {
        "resize_w": tgt_w, "resize_h": tgt_h,
        "offset_x": 0, "offset_y": 0,
    }
    return resized, crop_params


def scale_intrinsic_with_crop(K, orig_w, orig_h, crop_params):
    """Adjust a 3x3 intrinsic matrix for a resize-then-center-crop transform.

    Steps:
      1. Scale fx, fy, cx, cy for the resize.
      2. Shift cx, cy by the crop offset.
    """
    sx = crop_params["resize_w"] / orig_w
    sy = crop_params["resize_h"] / orig_h

    K_out = K.copy()
    K_out[0, 0] *= sx
    K_out[0, 2] = K[0, 2] * sx - crop_params["offset_x"]
    K_out[1, 1] *= sy
    K_out[1, 2] = K[1, 2] * sy - crop_params["offset_y"]
    return K_out


def preprocess_images_and_intrinsics(images, intrinsics, orig_wh, height, width, resize_mode="crop"):
    """Resize images and adjust intrinsics to model resolution.

    Args:
        images: List of PIL.Image (M context images).
        intrinsics: (N, 3, 3) numpy array at original resolution.
        orig_wh: (N, 2) array of (width, height) per frame, or a single
                 (width, height) tuple applied to all.
        height, width: Model resolution.
        resize_mode: "crop" or "stretch".

    Returns:
        resized_images: List of PIL.Image at (width, height).
        adjusted_intrinsics: (N, 3, 3) numpy array at model resolution.
    """
    resize_fn = resize_and_center_crop if resize_mode == "crop" else resize_stretch

    num_context = len(images)
    N = intrinsics.shape[0]

    if isinstance(orig_wh, (list, tuple)) and len(orig_wh) == 2 and not isinstance(orig_wh[0], (list, tuple, np.ndarray)):
        orig_wh_arr = np.array([orig_wh] * N, dtype=np.int32)
    else:
        orig_wh_arr = np.array(orig_wh, dtype=np.int32)
        if orig_wh_arr.shape[0] < N:
            pad = np.tile(orig_wh_arr[:1], (N - orig_wh_arr.shape[0], 1))
            orig_wh_arr = np.concatenate([orig_wh_arr, pad], axis=0)

    resized_images = []
    adjusted_K = intrinsics.copy()

    for i in range(num_context):
        img_np = np.array(images[i])
        ow, oh = orig_wh_arr[i]
        img_resized, crop_params = resize_fn(img_np, height, width)
        resized_images.append(Image.fromarray(img_resized))
        adjusted_K[i] = scale_intrinsic_with_crop(intrinsics[i], int(ow), int(oh), crop_params)

    for i in range(num_context, N):
        ow, oh = orig_wh_arr[i]
        _, crop_params = resize_fn(np.zeros((int(oh), int(ow), 3), dtype=np.uint8), height, width)
        adjusted_K[i] = scale_intrinsic_with_crop(intrinsics[i], int(ow), int(oh), crop_params)

    return resized_images, adjusted_K


# ---------------------------------------------------------------------------
# FrameCrafter model
# ---------------------------------------------------------------------------

class FrameCrafter:
    """Novel View Synthesis model built on Wan2.1.

    Handles pipeline loading, channel modification for separated encoding,
    and LoRA/checkpoint loading in one step.
    """

    IN_DIM = 420

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        vram_limit: float = 20,
        base_model_dir: str = None,
    ):
        """Load the model and checkpoint.

        Args:
            checkpoint_path: Path to .safetensors checkpoint (LoRA or full).
            device: Device to run on.
            vram_limit: GPU VRAM limit in GB for offloading (default 20
                for safe operation on 24 GB cards).
            base_model_dir: Directory containing pre-downloaded backbone
                weights (DiffSynth-Studio model layout). Set to None to
                fall back to model_id-based download.
        """
        self.device = device

        offload_kwargs = dict(
            offload_device="cpu",
            offload_dtype=torch.bfloat16,
            onload_device="cpu",
            onload_dtype=torch.bfloat16,
            preparing_device=device,
            preparing_dtype=torch.bfloat16,
            computation_device=device,
            computation_dtype=torch.bfloat16,
        )

        if base_model_dir is not None:
            dit_dir = os.path.join(base_model_dir, "Wan-AI/Wan2.1-I2V-14B-480P")
            dit_files = sorted(_glob.glob(os.path.join(dit_dir, "diffusion_pytorch_model*.safetensors")))

            # Resolve T5/VAE/CLIP from whichever layout is present:
            #   - ModelScope:   DiffSynth-Studio/Wan-Series-Converted-Safetensors/*.safetensors
            #   - HuggingFace:  Wan-AI/Wan2.1-I2V-14B-480P/*.pth
            converted_dir = os.path.join(base_model_dir, "DiffSynth-Studio/Wan-Series-Converted-Safetensors")
            def _resolve(name_safetensors: str, name_pth: str, label: str) -> str:
                converted = os.path.join(converted_dir, name_safetensors)
                original = os.path.join(dit_dir, name_pth)
                if os.path.exists(converted):
                    return converted
                if os.path.exists(original):
                    return original
                raise FileNotFoundError(
                    f"Could not find {label} weights under {base_model_dir}. "
                    f"Looked for:\n  {converted}\n  {original}"
                )

            t5_path = _resolve(
                "models_t5_umt5-xxl-enc-bf16.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "T5 text encoder",
            )
            vae_path = _resolve(
                "Wan2.1_VAE.safetensors",
                "Wan2.1_VAE.pth",
                "VAE",
            )
            clip_path = _resolve(
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "CLIP image encoder",
            )
            tokenizer_path = os.path.join(base_model_dir, "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")

            model_configs = [
                ModelConfig(path=dit_files, **offload_kwargs),
                ModelConfig(path=t5_path, **offload_kwargs),
                ModelConfig(path=vae_path, **offload_kwargs),
                ModelConfig(path=clip_path, **offload_kwargs),
            ]
            tokenizer_config = ModelConfig(path=tokenizer_path)
        else:
            model_configs = [
                ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors", **offload_kwargs),
                ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **offload_kwargs),
                ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth", **offload_kwargs),
                ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **offload_kwargs),
            ]
            tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

        # `redirect_common_files` rewrites the T5/VAE/CLIP file ids to
        # `DiffSynth-Studio/Wan-Series-Converted-Safetensors`, which only
        # exists on ModelScope. Disable the redirect when downloading from
        # HuggingFace so the original .pth files are pulled from the
        # `Wan-AI/Wan2.1-I2V-14B-480P` repo instead.
        download_source = os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
        redirect_common_files = download_source.lower() != "huggingface"

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=redirect_common_files,
            vram_limit=vram_limit,
        )
        self._modify_channels(self.IN_DIM)
        self._load_checkpoint(checkpoint_path)

    def _modify_channels(self, new_in_dim: int):
        """Replace patch_embedding to match training input dimension.

        The new Conv3d is re-wrapped with AutoWrappedModule so it participates
        in VRAM offloading.
        """
        model = self.pipe.dit
        if model is None:
            return

        old_in_dim = model.in_dim
        print(f"Modifying DiT input channels: {old_in_dim} -> {new_in_dim}")

        old_pe = model.patch_embedding
        pe_device = next(old_pe.parameters()).device
        pe_dtype = next(old_pe.parameters()).dtype
        new_pe = nn.Conv3d(new_in_dim, model.dim,
                           kernel_size=model.patch_size, stride=model.patch_size)
        new_pe = new_pe.to(device=pe_device, dtype=pe_dtype)

        sibling = None
        for child in model.children():
            if isinstance(child, AutoWrappedModule):
                sibling = child
                break

        if sibling is not None:
            wrapped_pe = AutoWrappedModule(
                new_pe,
                offload_dtype=sibling.offload_dtype,
                offload_device=sibling.offload_device,
                onload_dtype=sibling.onload_dtype,
                onload_device=sibling.onload_device,
                preparing_dtype=sibling.preparing_dtype,
                preparing_device=sibling.preparing_device,
                computation_dtype=sibling.computation_dtype,
                computation_device=sibling.computation_device,
                vram_limit=sibling.vram_limit,
                name="patch_embedding",
            )
            model.patch_embedding = wrapped_pe
        else:
            model.patch_embedding = new_pe

        del old_pe

        model.in_dim = new_in_dim
        model.individual_encoding = True

        torch.cuda.empty_cache()
        gc.collect()

        self.pipe.dit = model
        print("DiT channels modified successfully")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load LoRA or full checkpoint into the DiT.

        Tensors are loaded to CPU so the VRAM management system can move them
        to GPU layer-by-layer, avoiding OOM on memory-constrained cards.
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_state_dict(checkpoint_path, torch_dtype=torch.bfloat16, device="cpu")

        has_lora_keys = any(
            "lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k
            for k in checkpoint.keys()
        )

        if has_lora_keys:
            self.pipe.load_lora(self.pipe.dit, state_dict=checkpoint, alpha=1.0)
            print("LoRA weights loaded")

            patch_emb_state = {k: v for k, v in checkpoint.items() if "patch_embedding" in k}
            if patch_emb_state:
                pe = self.pipe.dit.patch_embedding
                if hasattr(pe, "module"):
                    inner_state = {k.replace("patch_embedding.", ""): v
                                   for k, v in patch_emb_state.items()}
                    pe.module.load_state_dict(inner_state, strict=False)
                else:
                    self.pipe.dit.load_state_dict(patch_emb_state, strict=False)
                print(f"Loaded {len(patch_emb_state)} patch_embedding parameters")
            else:
                print("Warning: No patch_embedding weights found in checkpoint!")
        else:
            load_result = self.pipe.dit.load_state_dict(checkpoint, strict=False)
            missing = [k for k in load_result.missing_keys if k not in checkpoint]
            unexpected = load_result.unexpected_keys
            print(f"Full checkpoint loaded -- {len(checkpoint)} keys")
            if missing:
                print(f"  Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    def generate(
        self,
        images: list,
        w2c_poses: np.ndarray,
        intrinsics: np.ndarray,
        height: int = 192,
        width: int = 336,
        num_inference_steps: int = 50,
        seed: int = 42,
        cfg_scale: float = 1.0,
        prompt: str = "",
        resize_mode: str = None,
        orig_wh=None,
    ) -> list:
        """Generate novel views from context images and camera parameters.

        Args:
            images: List of PIL.Image context views (M images).
            w2c_poses: (N, 4, 4) numpy array -- w2c poses in OpenCV convention
                       (x-right, y-down, z-forward) for ALL frames
                       (M context + K target).
                       Order: context first, targets last.
            intrinsics: (N, 3, 3) numpy array -- intrinsics at original
                        resolution (if resize_mode is set) or at model
                        resolution (if resize_mode is None).
            height: Model input height (must match training resolution).
            width: Model input width (must match training resolution).
            num_inference_steps: Number of diffusion steps.
            seed: Random seed.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            prompt: Text prompt (usually empty for NVS).
            resize_mode: "crop" (resize + center-crop) or "stretch" (plain
                         resize). If None, images and intrinsics are assumed
                         to already be at model resolution.
            orig_wh: Override for the original image dimensions used to
                     scale intrinsics. Defaults to the first image's size.
                     Only needed if intrinsics correspond to a different
                     resolution than the input images.

        Returns:
            List of PIL.Image -- all frames (context + generated targets).
            The last K frames are the generated novel views.
        """
        if resize_mode is not None:
            if orig_wh is None:
                orig_wh = (images[0].width, images[0].height)
            images, intrinsics = preprocess_images_and_intrinsics(
                images, intrinsics, orig_wh, height, width, resize_mode
            )

        num_total = w2c_poses.shape[0]

        raymap = prepare_raymap(w2c_poses, intrinsics, height, width)
        raymap = raymap.to(self.device, dtype=torch.bfloat16)

        video = self.pipe(
            prompt=prompt,
            negative_prompt="",
            input_image=images,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_total,
            num_latent_frames=num_total,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            tiled=True,
        )
        return video
