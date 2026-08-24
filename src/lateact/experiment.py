"""Matrix-Game loading, conditions, decoding, and motion metrics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lateact.runtime import tensor_sha256


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scene_path(code_root: Path, index: int) -> Path:
    suffix = "webp" if index == 10 else "png"
    return code_root / f"demo_images/universal/{index:04d}.{suffix}"


def resize_crop(image, target_h: int, target_w: int):
    width, height = image.size
    if height / width > target_h / target_w:
        new_width = width
        new_height = int(new_width * target_h / target_w)
    else:
        new_height = height
        new_width = int(new_height * target_w / target_h)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    return image.crop((left, top, left + new_width, top + new_height))


def load_models(upstream: Path, model_root: Path, device: torch.device):
    code_root = upstream.resolve() / "Matrix-Game-2"
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))

    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from safetensors.torch import load_file
    from utils.wan_wrapper import WanDiffusionWrapper
    from wan.vae.wanx_vae import get_wanx_vae_wrapper

    config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")
    generator = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)
    generator.load_state_dict(
        load_file(str(model_root.resolve() / "base_distill.safetensors"), device="cpu")
    )
    pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=None)
    pipeline.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    vae = get_wanx_vae_wrapper(str(model_root.resolve()), torch.float16)
    vae.to(device, torch.bfloat16)
    return code_root, pipeline, vae


@torch.no_grad()
def prepare_scene(image_path: Path, total_latents: int, vae, device) -> tuple[dict, dict]:
    from diffusers.utils import load_image
    from torchvision.transforms import v2

    image = resize_crop(load_image(str(image_path)), 352, 640)
    transform = v2.Compose(
        [
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    image_tensor = transform(image)[None, :, None].to(
        device=device, dtype=torch.bfloat16
    )
    raw_frames = (total_latents - 1) * 4 + 1
    padding = torch.zeros_like(image_tensor).repeat(1, 1, raw_frames - 1, 1, 1)
    encoded = vae.encode(
        torch.cat([image_tensor, padding], dim=2),
        device=device,
        tiled=True,
        tile_size=[44, 80],
        tile_stride=[23, 38],
    ).to(device=device, dtype=torch.bfloat16)
    mask = torch.ones_like(encoded)
    mask[:, :, 1:] = 0
    fixed = {
        "cond_concat": torch.cat([mask[:, :4], encoded], dim=1),
        "visual_context": vae.clip.encode_video(image_tensor).to(
            device=device, dtype=torch.bfloat16
        ),
    }
    hashes = {name: tensor_sha256(value) for name, value in fixed.items()}
    return fixed, hashes


def make_condition(
    fixed: dict,
    *,
    raw_frames: int,
    branch_start_raw: int,
    action: str,
    device,
) -> dict:
    keyboard = torch.zeros((1, raw_frames, 4), dtype=torch.bfloat16, device=device)
    mouse = torch.zeros((1, raw_frames, 2), dtype=torch.bfloat16, device=device)
    if action == "mouse_left":
        mouse[:, branch_start_raw:, 1] = -0.1
    elif action == "mouse_right":
        mouse[:, branch_start_raw:, 1] = 0.1
    elif action == "keyboard_left":
        keyboard[:, branch_start_raw:, 2] = 1
    elif action == "keyboard_right":
        keyboard[:, branch_start_raw:, 3] = 1
    elif action != "neutral":
        raise ValueError(action)
    return {
        "cond_concat": fixed["cond_concat"],
        "visual_context": fixed["visual_context"],
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }


def make_transition_condition(
    fixed: dict,
    *,
    raw_frames: int,
    first_start_raw: int,
    second_start_raw: int,
    first_action: str,
    second_action: str,
    device,
) -> dict:
    """Condition with one action for the current block and another for the next."""
    result = make_condition(
        fixed,
        raw_frames=raw_frames,
        branch_start_raw=first_start_raw,
        action=first_action,
        device=device,
    )
    replacement = make_condition(
        fixed,
        raw_frames=raw_frames,
        branch_start_raw=second_start_raw,
        action=second_action,
        device=device,
    )
    result["mouse_cond"][:, second_start_raw:] = replacement["mouse_cond"][:, second_start_raw:]
    result["keyboard_cond"][:, second_start_raw:] = replacement["keyboard_cond"][:, second_start_raw:]
    return result


def condition_hashes(condition: dict) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in condition.items()}


def decoded_uint8(video: torch.Tensor) -> np.ndarray:
    return (
        video[0]
        .permute(1, 2, 3, 0)
        .float()
        .add(1)
        .mul(127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def save_video(path: Path, frames: np.ndarray) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, frames, fps=12, quality=8)


def signed_camera_motion(frames: np.ndarray, prefix_frames: int) -> dict:
    """Robust signed horizontal flow from the prefix boundary onward."""
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (320, 176))
        for frame in frames[prefix_frames - 1 :]
    ]
    lk_dx: list[float] = []
    dense_dx: list[float] = []
    track_counts: list[int] = []
    for previous, current in zip(gray, gray[1:]):
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=600, qualityLevel=0.01, minDistance=5, blockSize=7
        )
        valid_dx = np.empty(0, dtype=np.float32)
        if points is not None:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if moved is not None and status is not None:
                delta = (moved - points).reshape(-1, 2)
                valid = status.reshape(-1).astype(bool) & np.isfinite(delta).all(axis=1)
                valid_dx = delta[valid, 0]
                if valid_dx.size:
                    median = np.median(valid_dx)
                    mad = np.median(np.abs(valid_dx - median))
                    if mad > 0:
                        valid_dx = valid_dx[np.abs(valid_dx - median) <= 3.5 * 1.4826 * mad]
        track_counts.append(int(valid_dx.size))
        lk_dx.append(float(np.median(valid_dx)) if valid_dx.size else float("nan"))

        dense = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        dense_dx.append(float(np.median(dense[..., 0])))

    return {
        "lk_signed_sum": float(np.nansum(lk_dx)),
        "dense_signed_sum": float(np.sum(dense_dx)),
        "lk_per_transition": lk_dx,
        "dense_per_transition": dense_dx,
        "track_counts": track_counts,
        "median_track_count": float(np.median(track_counts)) if track_counts else 0.0,
    }


def signed_lateral_translation(frames: np.ndarray, prefix_frames: int) -> dict:
    """Robust signed global scene translation for keyboard A/D locomotion.

    A forward/backward Lucas-Kanade check rejects unstable tracks. A partial
    affine RANSAC fit then measures the horizontal displacement of the image
    center, avoiding coordinate-origin artifacts from small fitted rotations.
    The independent median-track sum is retained only as a sign-stability audit.
    """
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (320, 176))
        for frame in frames[prefix_frames - 1 :]
    ]
    affine_dx: list[float] = []
    median_dx: list[float] = []
    tracked_counts: list[int] = []
    inlier_counts: list[int] = []
    center = np.array([160.0, 88.0, 1.0], dtype=np.float64)
    for previous, current in zip(gray, gray[1:]):
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=800, qualityLevel=0.01, minDistance=5, blockSize=7
        )
        if points is None:
            affine_dx.append(float("nan"))
            median_dx.append(float("nan"))
            tracked_counts.append(0)
            inlier_counts.append(0)
            continue
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if moved is None or status is None:
            affine_dx.append(float("nan"))
            median_dx.append(float("nan"))
            tracked_counts.append(0)
            inlier_counts.append(0)
            continue
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            moved,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or backward_status is None:
            affine_dx.append(float("nan"))
            median_dx.append(float("nan"))
            tracked_counts.append(0)
            inlier_counts.append(0)
            continue
        forward_ok = status.reshape(-1).astype(bool)
        backward_ok = backward_status.reshape(-1).astype(bool)
        round_trip = np.linalg.norm((backward - points).reshape(-1, 2), axis=1)
        valid = forward_ok & backward_ok & np.isfinite(round_trip) & (round_trip < 1.5)
        first = points.reshape(-1, 2)[valid]
        second = moved.reshape(-1, 2)[valid]
        tracked_counts.append(int(len(first)))
        median_dx.append(
            float(np.median(second[:, 0] - first[:, 0]))
            if len(first)
            else float("nan")
        )
        matrix, inliers = (None, None)
        if len(first) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                first,
                second,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
        if matrix is None or inliers is None:
            affine_dx.append(float("nan"))
            inlier_counts.append(0)
        else:
            affine_dx.append(float((matrix @ center)[0] - center[0]))
            inlier_counts.append(int(inliers.sum()))
    return {
        "affine_center_signed_sum": float(np.nansum(affine_dx)),
        "median_lk_signed_sum": float(np.nansum(median_dx)),
        "affine_per_transition": affine_dx,
        "median_lk_per_transition": median_dx,
        "tracked_counts": tracked_counts,
        "inlier_counts": inlier_counts,
        "median_inlier_count": float(np.median(inlier_counts)) if inlier_counts else 0.0,
    }
