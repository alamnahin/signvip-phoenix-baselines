#!/usr/bin/env python3
"""Render recorded SignViP reproduction recipes with run-specific paths."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=(
            "fsq",
            "extract",
            "gpt",
            "vq-eval",
            "t2s-eval",
            "continuous-eval",
            "oracle-eval",
        ),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-encoder", type=Path)
    parser.add_argument("--vq-checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--token-root", type=Path)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--dwpose-root", type=Path)
    parser.add_argument("--hamer-root", type=Path)
    args = parser.parse_args()

    config = OmegaConf.load(args.input)
    if args.kind == "fsq":
        if args.base_encoder is None:
            parser.error("--base-encoder is required for fsq")
        config.modules.condition_encoder = str(args.base_encoder)
        config.modules.condition_encoder_motion = str(args.base_encoder)
    elif args.kind in ("extract", "vq-eval"):
        if args.vq_checkpoint is None:
            parser.error("--vq-checkpoint is required")
        config.modules.vq_model = str(args.vq_checkpoint)
        if args.metadata is not None:
            config.dataset.meta_paths = [str(args.metadata)]
        if args.kind == "extract":
            if args.rgb_root is not None:
                config.dataset.roots = [str(args.rgb_root)]
            if args.dwpose_root is not None:
                config.dataset.sk_roots = [str(args.dwpose_root)]
            if args.hamer_root is not None:
                config.dataset.hamer_roots = [str(args.hamer_root)]
    elif args.kind == "gpt":
        if args.token_root is None:
            parser.error("--token-root is required for gpt")
        # Full split metadata already includes train_processed_videos/ or
        # dev_processed_videos/ in each relative video path.
        config.dataset.pose_roots = [str(args.token_root)]
        config.validation_data.pose_roots = [str(args.token_root)]
    elif args.kind == "t2s-eval":
        # Text-to-video evaluation does not consume ground-truth FSQ tokens.
        # Phoenix test pose inputs are not part of the local dataset export.
        config.dataset.pose_roots = None
    elif args.kind == "continuous-eval":
        required = {
            "base-encoder": args.base_encoder,
            "metadata": args.metadata,
            "rgb-root": args.rgb_root,
            "dwpose-root": args.dwpose_root,
            "hamer-root": args.hamer_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "continuous-eval requires " + ", ".join(f"--{name}" for name in missing)
            )
        config.modules.condition_encoder = str(args.base_encoder)
        config.modules.condition_encoder_motion = str(args.base_encoder)
        config.modules.vq_model = None
        config.modules.condition_encoder_kwargs = {"use_vq": False}
        config.dataset.roots = [str(args.rgb_root)]
        config.dataset.sk_roots = [str(args.dwpose_root)]
        config.dataset.hamer_roots = [str(args.hamer_root)]
        config.dataset.meta_paths = [str(args.metadata)]
        config.dataset.frame_scale = [1.0, 1.0]
        config.dataset.sample_rate = 1
    else:
        required = {
            "metadata": args.metadata,
            "token-root": args.token_root,
            "rgb-root": args.rgb_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "oracle-eval requires " + ", ".join(f"--{name}" for name in missing)
            )
        config.dataset.roots = [str(args.rgb_root)]
        config.dataset.pose_roots = [str(args.token_root)]
        config.dataset.meta_paths = [str(args.metadata)]
        config.dataset.sample_rate = 1
        config.eval.batch_size = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)


if __name__ == "__main__":
    main()
