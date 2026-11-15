#!/usr/bin/env python3
"""Prepare a fixed Phoenix2014T subset shared by both generator evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--recognition-info", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "dev", "test"), default="test"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--subset-metadata", type=Path, required=True)
    parser.add_argument("--eval-spec", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.limit <= 0 or args.limit > len(rows):
        raise ValueError(f"limit must be in [1, {len(rows)}], got {args.limit}")

    info = np.load(args.recognition_info, allow_pickle=True).item()
    recognition_labels = {
        value["fileid"]: value["label"].strip()
        for key, value in info.items()
        if key != "prefix"
    }
    subset_rows = []
    samples = []
    for index, row in enumerate(rows[: args.limit]):
        video_name = Path(row["video"]).name
        source_path = (args.video_root / video_name).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source video: {source_path}")
        sample_id = Path(video_name).stem
        reference_gloss = recognition_labels.get(sample_id)
        if not reference_gloss:
            raise KeyError(f"No canonical recognition label for {sample_id}")

        normalized = dict(row)
        normalized["video"] = video_name
        subset_rows.append(normalized)
        samples.append(
            {
                "rank": index,
                "id": sample_id,
                "video": video_name,
                "text": row["text"].strip(),
                "prompt_text": row["text"].strip(),
                "reference_gloss": reference_gloss,
                "source_path": str(source_path),
                "generation_seed": 42 + index,
            }
        )

    args.subset_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.eval_spec.parent.mkdir(parents=True, exist_ok=True)
    args.subset_metadata.write_text(
        json.dumps(subset_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "protocol": f"phoenix2014t-{args.split}-first{len(samples)}-v1",
        "split": args.split,
        "selection": (
            f"first entries in the shared Phoenix2014T {args.split} metadata"
        ),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": sha256(args.metadata),
        "recognition_info": str(args.recognition_info.resolve()),
        "recognition_info_sha256": sha256(args.recognition_info),
        "num_samples": len(samples),
        "samples": samples,
    }
    args.eval_spec.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(samples)} Phoenix2014T {args.split} samples.")


if __name__ == "__main__":
    main()
