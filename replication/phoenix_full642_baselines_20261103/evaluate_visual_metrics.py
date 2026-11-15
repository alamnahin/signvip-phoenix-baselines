#!/usr/bin/env python3
"""Evaluate available SignViP table metrics under a recorded local protocol."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

SIGNVIP_ROOT = Path('/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip')
sys.path.insert(0, str(SIGNVIP_ROOT))

from metrics.calculate_fid import InceptionV3
from metrics.calculate_ssim import calculate_ssim_function
from metrics.fvd.styleganv.fvd import get_fvd_feats, load_i3d_pretrained


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-spec", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def read_uniform(path, frame_count, size=(224, 224)):
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise RuntimeError(f"No decodable frames in {path}")
    indices = np.linspace(0, total - 1, frame_count).round().astype(np.int64)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    capture.release()
    return np.stack(frames).astype(np.float32) / 255.0


def frechet(first, second, epsilon=1e-6):
    first_mean, second_mean = first.mean(0), second.mean(0)
    first_cov = np.cov(first, rowvar=False)
    second_cov = np.cov(second, rowvar=False)
    delta = first_mean - second_mean
    covariance_mean, _ = linalg.sqrtm(first_cov @ second_cov, disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(first_cov.shape[0]) * epsilon
        covariance_mean = linalg.sqrtm(
            (first_cov + offset) @ (second_cov + offset), disp=False
        )[0]
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(
        delta @ delta
        + np.trace(first_cov)
        + np.trace(second_cov)
        - 2 * np.trace(covariance_mean)
    )


@torch.inference_mode()
def inception_features(model, clips, device, batch_size):
    frames = clips.reshape(-1, *clips.shape[2:])
    features = []
    for start in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[start : start + batch_size]).permute(0, 3, 1, 2)
        output = model(batch.to(device))[0]
        if output.shape[-2:] != (1, 1):
            output = F.adaptive_avg_pool2d(output, (1, 1))
        features.append(output.flatten(1).float().cpu().numpy())
    return np.concatenate(features)


@torch.inference_mode()
def i3d_features(model, clips, device, batch_size):
    values = []
    for start in range(0, len(clips), batch_size):
        batch = torch.from_numpy(clips[start : start + batch_size]).permute(0, 4, 1, 2, 3)
        values.append(get_fvd_feats(batch, i3d=model, device=device, bs=batch_size))
    return np.concatenate(values)


@torch.inference_mode()
def clip_features(model, processor, clips, device, batch_size):
    frames = clips.reshape(-1, *clips.shape[2:])
    features = []
    for start in range(0, len(frames), batch_size):
        images = [
            (frame * 255).round().clip(0, 255).astype(np.uint8)
            for frame in frames[start : start + batch_size]
        ]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        features.append(model(pixel_values=pixels).image_embeds.float().cpu().numpy())
    return np.concatenate(features)


@torch.inference_mode()
def paired_metrics(lpips_model, generated, target, device, batch_size):
    generated_frames = generated.reshape(-1, *generated.shape[2:])
    target_frames = target.reshape(-1, *target.shape[2:])
    ssim_values = []
    psnr_values = []
    lpips_values = []
    for start in tqdm(range(0, len(generated_frames), batch_size), desc="Paired metrics"):
        generated_batch = torch.from_numpy(generated_frames[start : start + batch_size]).permute(0, 3, 1, 2)
        target_batch = torch.from_numpy(target_frames[start : start + batch_size]).permute(0, 3, 1, 2)
        mse = (generated_batch - target_batch).square().flatten(1).mean(1)
        psnr_values.extend((-10 * torch.log10(mse.clamp_min(1e-12))).tolist())
        for generated_frame, target_frame in zip(generated_batch, target_batch):
            ssim_values.append(
                calculate_ssim_function(
                    generated_frame.numpy(), target_frame.numpy()
                )
            )
        distance = lpips_model(
            generated_batch.to(device).mul(2).sub(1),
            target_batch.to(device).mul(2).sub(1),
        )
        lpips_values.extend(distance.flatten().float().cpu().tolist())
    return {
        "ssim": float(np.mean(ssim_values)),
        "psnr_db": float(np.mean(psnr_values)),
        "lpips_alex": float(np.mean(lpips_values)),
    }


def main():
    args = parse_args()
    spec = json.loads(args.eval_spec.read_text(encoding="utf-8"))
    pairs = []
    missing = []
    for sample in spec["samples"]:
        generated = args.video_dir / sample["video"]
        target = Path(sample["source_path"])
        if generated.is_file():
            pairs.append((sample["id"], generated, target))
        else:
            missing.append(sample["video"])
    if len(pairs) < 2:
        raise RuntimeError(f"Only {len(pairs)} generated videos are available")

    generated_clips = []
    target_clips = []
    for _, generated, target in tqdm(pairs, desc="Loading aligned clips"):
        generated_clips.append(read_uniform(generated, args.num_frames))
        target_clips.append(read_uniform(target, args.num_frames))
    generated_clips = np.stack(generated_clips)
    target_clips = np.stack(target_clips)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    i3d = load_i3d_pretrained(device=device)
    inception = InceptionV3([InceptionV3.DEFAULT_BLOCK_INDEX]).eval().to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    target_i3d = i3d_features(i3d, target_clips, device, args.batch_size)
    generated_i3d = i3d_features(i3d, generated_clips, device, args.batch_size)
    target_inception = inception_features(
        inception, target_clips, device, args.batch_size
    )
    generated_inception = inception_features(
        inception, generated_clips, device, args.batch_size
    )
    paired = paired_metrics(
        lpips_model, generated_clips, target_clips, device, args.batch_size
    )

    clip_fid = None
    clip_reason = None
    try:
        clip_name = "openai/clip-vit-base-patch32"
        clip_processor = CLIPImageProcessor.from_pretrained(
            clip_name, local_files_only=True
        )
        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            clip_name, local_files_only=True
        ).eval().to(device)
        target_clip = clip_features(
            clip_model, clip_processor, target_clips, device, args.batch_size
        )
        generated_clip = clip_features(
            clip_model, clip_processor, generated_clips, device, args.batch_size
        )
        clip_fid = frechet(generated_clip, target_clip)
    except Exception as error:
        clip_reason = (
            "OpenAI CLIP ViT-B/32 was not completely cached at evaluation time: "
            f"{type(error).__name__}: {error}"
        )

    unavailable = {
        "clip_fid": {
            "value": clip_fid,
            "reason": clip_reason,
        },
        "identity_similarity": {
            "value": None,
            "reason": "The paper's identity detector/embedding protocol and checkpoint are unavailable.",
        },
        "hand_ssim": {
            "value": None,
            "reason": "The paper's hand detector/cropping protocol and checkpoint are unavailable.",
        },
        "comet": {
            "value": None,
            "reason": "CorrNet+ predicts glosses, not spoken-language translations required by COMET.",
        },
    }
    result = {
        "protocol": {
            "eval_spec": str(args.eval_spec.resolve()),
            "requested_videos": len(spec["samples"]),
            "evaluated_videos": len(pairs),
            "missing_videos": missing,
            "temporal_alignment": (
                f"independent uniform sampling of {args.num_frames} frames over each "
                "generated and ground-truth clip"
            ),
            "spatial_alignment": "each sampled RGB frame resized to 224x224",
            "fvd": "SignViP repository StyleGAN-V I3D TorchScript features",
            "fid": "SignViP repository ImageNet Inception-v3 pool3 features",
            "clip_fid": (
                "OpenAI CLIP ViT-B/32 projected image embeddings (unnormalized)"
            ),
            "lpips": "LPIPS 0.1.4, AlexNet backbone",
            "paper_comparable": False,
            "noncomparability_reason": (
                "full 642-video test split and explicit local alignment; the paper does "
                "not release complete preprocessing/evaluator details"
            ),
        },
        "metrics": {
            "fvd_i3d": frechet(generated_i3d, target_i3d),
            "fid_inception": frechet(generated_inception, target_inception),
            **paired,
        },
        "runtime": {
            "lpips_version": importlib.metadata.version("lpips"),
            "torch_version": torch.__version__,
        },
        "unavailable": unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
