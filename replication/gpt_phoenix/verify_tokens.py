#!/usr/bin/env python3
"""Verify completeness and FSQ contract for extracted Phoenix tokens."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--meta", type=Path, action="append", required=True)
    parser.add_argument("--pose-size", type=int, default=12)
    parser.add_argument("--codebook-size", type=int, default=625)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    expected = []
    for meta_path in args.meta:
        with meta_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        expected.extend(
            (args.token_root / row["video"]).with_suffix(".pkl") for row in rows
        )

    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} token files. First paths:\n"
            + "\n".join(missing[:20])
        )

    active_codes = np.zeros(args.codebook_size, dtype=np.int64)
    total_frames = 0
    minimum = args.codebook_size
    maximum = -1
    for path in expected:
        with path.open("rb") as handle:
            tokens = np.asarray(pickle.load(handle))
        if tokens.ndim != 2 or tokens.shape[1] != args.pose_size or not len(tokens):
            raise ValueError(
                f"Invalid token shape for {path}: {tokens.shape}; expected [T, {args.pose_size}]"
            )
        if not np.all(np.isfinite(tokens)) or not np.all(tokens == tokens.astype(np.int64)):
            raise ValueError(f"Non-integer or non-finite tokens in {path}")
        tokens = tokens.astype(np.int64, copy=False)
        token_min = int(tokens.min())
        token_max = int(tokens.max())
        if token_min < 0 or token_max >= args.codebook_size:
            raise ValueError(
                f"Out-of-range token in {path}: [{token_min}, {token_max}]"
            )
        active_codes += np.bincount(tokens.ravel(), minlength=args.codebook_size)
        total_frames += len(tokens)
        minimum = min(minimum, token_min)
        maximum = max(maximum, token_max)

    probabilities = active_codes[active_codes > 0] / active_codes.sum()
    perplexity = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    report = {
        "clips": len(expected),
        "frames": total_frames,
        "pose_size": args.pose_size,
        "codebook_size": args.codebook_size,
        "minimum_token": minimum,
        "maximum_token": maximum,
        "active_codes": int(np.count_nonzero(active_codes)),
        "token_perplexity": perplexity,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
