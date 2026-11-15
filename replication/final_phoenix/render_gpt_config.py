#!/usr/bin/env python3
"""Render the final GPT recipe with run-specific token and FSQ paths."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--fsq-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.input)
    config.dataset.pose_roots = [str(args.token_root)]
    config.validation_data.pose_roots = [str(args.token_root)]
    config.validation_metric.checkpoint = str(args.fsq_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)


if __name__ == "__main__":
    main()

