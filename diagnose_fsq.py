import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder
from signdatasets import SignLangVideoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure held-out FSQ reconstruction without diffusion or GPT."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--vq-checkpoint")
    parser.add_argument("--metadata", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-frames", type=int, default=36)
    parser.add_argument("--sample-rate", type=int, default=1)
    return parser.parse_args()


def load_encoders(cfg, device):
    common = dict(
        conditioning_channels=3,
        image_finetune=False,
        num_conds=2,
        motion_module_type=cfg.modules.unet_additional_kwargs.motion_module_type,
        motion_module_kwargs=cfg.modules.unet_additional_kwargs.motion_module_kwargs,
    )
    vq_kwargs = OmegaConf.to_container(
        cfg.modules.condition_encoder_kwargs, resolve=True
    )
    continuous_kwargs = dict(vq_kwargs)
    continuous_kwargs["use_vq"] = False

    quantized = VQConditionEncoder(**common, **vq_kwargs)
    continuous = VQConditionEncoder(**common, **continuous_kwargs)

    base_state = torch.load(cfg.modules.condition_encoder, map_location="cpu")
    motion_state = torch.load(
        cfg.modules.condition_encoder_motion, map_location="cpu"
    )
    base_state.update(motion_state)
    continuous_state = {
        key: value for key, value in base_state.items() if not key.startswith("vq.")
    }
    continuous.load_state_dict(continuous_state, strict=True)

    full_state = dict(base_state)
    trained_vq_state = torch.load(cfg.modules.vq_model, map_location="cpu")
    trained_vq_state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in trained_vq_state.items()
    }
    full_state.update(trained_vq_state)
    missing, unexpected = quantized.load_state_dict(full_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    continuous.to(device=device, dtype=torch.float32).eval()
    quantized.to(device=device, dtype=torch.float32).eval()
    return continuous, quantized


def build_dataset(cfg, num_frames, sample_rate):
    return SignLangVideoDataset(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.get("frame_scale", [1.0, 1.0]),
        frame_ratio=cfg.dataset.get("frame_ratio", [0.9, 1.0]),
        roots=cfg.dataset.roots[0],
        sk_roots=cfg.dataset.sk_roots[0],
        hamer_roots=cfg.dataset.hamer_roots[0],
        meta_paths=cfg.validation_data.meta_paths,
        sample_rate=sample_rate,
        num_frames=num_frames,
        ref_margin=cfg.dataset.get("ref_margin", 1),
        uncond_ratio=0,
        mask_ratio=0,
        mask_thershold=0,
        skip_ratio=0,
        random_sample=False,
        sk_mask_ratio=0,
        hamer_mask_ratio=0,
        both_mask_ratio=0,
    )


def tensor_metrics(target, reconstructed):
    target = target.float()
    reconstructed = reconstructed.float()
    mse = F.mse_loss(reconstructed, target).item()
    energy = target.square().mean().item()
    cosine = F.cosine_similarity(
        target.flatten(1), reconstructed.flatten(1), dim=1
    ).mean().item()
    return {
        "mse": mse,
        "nmse": mse / max(energy, 1e-12),
        "cosine": cosine,
    }


def append_metrics(totals, prefix, metrics):
    for name, value in metrics.items():
        totals[f"{prefix}/{name}"].append(value)


@torch.inference_mode()
def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    if args.metadata:
        if "validation_data" not in cfg:
            cfg.validation_data = {}
        cfg.validation_data.meta_paths = args.metadata
    if args.vq_checkpoint:
        cfg.modules.vq_model = args.vq_checkpoint
    if not cfg.modules.get("vq_model"):
        raise ValueError("Set modules.vq_model in the config or pass --vq-checkpoint.")
    continuous, quantized = load_encoders(cfg, device)
    dataset = build_dataset(cfg, args.num_frames, args.sample_rate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    totals = defaultdict(list)
    token_change_rates = []
    target_motion_energies = []
    raw_motion_energies = []
    delta_error_sum = 0.0
    delta_target_sum = 0.0
    delta_reconstructed_sum = 0.0
    delta_dot_sum = 0.0
    code_counts = torch.zeros(
        quantized.vq.n_e, dtype=torch.long, device=device
    )
    samples = min(args.max_samples, len(dataset))

    for sample_index, batch in enumerate(
        tqdm(dataloader, total=samples, desc="Diagnosing FSQ")
    ):
        if sample_index >= samples:
            break
        sk = rearrange(
            batch["tgt_sk_frames"].to(device), "b c f h w -> (b f) c h w"
        )
        hamer = rearrange(
            batch["tgt_hamer_frames"].to(device), "b c f h w -> (b f) c h w"
        )
        frame_count = sk.shape[0]

        raw_condition = torch.cat((sk, hamer), dim=1)
        raw_delta = raw_condition[1:] - raw_condition[:-1]
        raw_motion_energies.extend(
            raw_delta.square().flatten(1).mean(dim=1).cpu().tolist()
        )

        target_mid = continuous.encode(sk, hamer)
        reconstructed_mid, _, _ = quantized.vq(target_mid)
        indices = quantized.vq.encode(target_mid)
        code_counts += torch.bincount(
            indices.reshape(-1), minlength=quantized.vq.n_e
        )

        append_metrics(
            totals, "mid", tensor_metrics(target_mid, reconstructed_mid)
        )
        target_delta = target_mid[1:] - target_mid[:-1]
        reconstructed_delta = reconstructed_mid[1:] - reconstructed_mid[:-1]
        append_metrics(
            totals,
            "mid_temporal_delta",
            tensor_metrics(target_delta, reconstructed_delta),
        )

        delta_error_sum += (reconstructed_delta - target_delta).square().sum().item()
        delta_target_sum += target_delta.square().sum().item()
        delta_reconstructed_sum += reconstructed_delta.square().sum().item()
        delta_dot_sum += (target_delta * reconstructed_delta).sum().item()

        frame_motion = target_delta.square().flatten(1).mean(dim=1)
        frame_token_change = (indices[1:] != indices[:-1]).float().mean(dim=1)
        target_motion_energies.extend(frame_motion.cpu().tolist())
        token_change_rates.extend(frame_token_change.cpu().tolist())

        target_outputs = continuous.decode(target_mid, video_length=frame_count)
        reconstructed_outputs = continuous.decode(
            reconstructed_mid, video_length=frame_count
        )
        for level, (target, reconstructed) in enumerate(
            zip(target_outputs, reconstructed_outputs)
        ):
            append_metrics(
                totals,
                f"condition_level_{level}",
                tensor_metrics(target, reconstructed),
            )

        del target_outputs, reconstructed_outputs
        del target_mid, reconstructed_mid, indices, raw_condition, raw_delta, sk, hamer

    probabilities = code_counts.float() / code_counts.sum().clamp_min(1)
    perplexity = torch.exp(
        -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
    )
    used_codes = int((code_counts > 0).sum().item())
    motion = np.asarray(target_motion_energies)
    raw_motion = np.asarray(raw_motion_energies)
    token_change = np.asarray(token_change_rates)
    motion_change_correlation = (
        float(np.corrcoef(motion, token_change)[0, 1])
        if len(motion) > 1 and motion.std() > 0 and token_change.std() > 0
        else None
    )
    quiet_threshold, active_threshold = np.quantile(motion, [0.25, 0.75])

    summary = {
        "checkpoint": str(cfg.modules.vq_model),
        "split_metadata": list(cfg.validation_data.meta_paths),
        "samples": samples,
        "frames_per_sample": args.num_frames,
        "sample_rate": args.sample_rate,
        "codebook_size": int(quantized.vq.n_e),
        "used_codes": used_codes,
        "usage_fraction": used_codes / quantized.vq.n_e,
        "perplexity": perplexity.item(),
        "temporal": {
            "global_delta_nmse": delta_error_sum / max(delta_target_sum, 1e-12),
            "global_delta_cosine": delta_dot_sum
            / max((delta_target_sum * delta_reconstructed_sum) ** 0.5, 1e-12),
            "mean_feature_motion_energy": float(motion.mean()),
            "mean_raw_condition_motion_energy": float(raw_motion.mean()),
            "encoded_to_raw_motion_ratio": float(motion.mean() / raw_motion.mean()),
            "mean_token_change_fraction": float(token_change.mean()),
            "quiet_quartile_token_change_fraction": float(
                token_change[motion <= quiet_threshold].mean()
            ),
            "active_quartile_token_change_fraction": float(
                token_change[motion >= active_threshold].mean()
            ),
            "feature_motion_token_change_correlation": motion_change_correlation,
        },
        "metrics": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for key, values in sorted(totals.items())
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
