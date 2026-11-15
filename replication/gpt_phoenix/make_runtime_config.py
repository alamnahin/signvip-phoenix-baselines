#!/usr/bin/env python3
"""Derive small runtime configs without modifying the recorded recipes."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("extract", "gpt", "paper", "corrected", "corrected-eval"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--attention")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    config = OmegaConf.load(args.input)
    if args.kind == "extract":
        if args.meta is None:
            parser.error("--meta is required for an extraction preflight")
        config.dataset.meta_paths = [str(args.meta)]
    elif args.kind == "gpt":
        config.solver.max_train_steps = 1
        config.checkpointing_steps = 2
        config.valid_steps = 2
        config.dataloader.batch_size = 2
        config.dataloader.num_workers = 0
    elif args.kind == "paper":
        if args.batch_size is None or args.batch_size < 1:
            parser.error("--batch-size must be positive for a paper config")
        config.dataloader.batch_size = args.batch_size
    elif args.kind == "corrected":
        if args.learning_rate is None or args.learning_rate <= 0:
            parser.error("--learning-rate must be positive for a corrected config")
        if args.attention not in ("global", "local:128"):
            parser.error("--attention must be global or local:128")
        config.solver.optimizer.learning_rate = args.learning_rate
        config.modules.attn_type = args.attention
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for corrected-eval")
        if args.attention not in ("global", "local:128"):
            parser.error("--attention must be global or local:128")
        config.modules.ckpt = str(args.checkpoint)
        config.modules.attn_type = args.attention

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)


if __name__ == "__main__":
    main()
