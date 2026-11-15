#!/usr/bin/env python3
"""Validate Phoenix inputs and create deterministic extraction shards."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-meta", type=Path, required=True)
    parser.add_argument("--dev-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_metadata(path):
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty metadata list: {path}")
    return rows


def condition_paths(dataset_root, row):
    video = Path(row["video"])
    hamer_video = Path(str(video).replace("processed_videos", "processed_videos2"))
    return (
        dataset_root / "rgb" / video,
        (dataset_root / "dwpose" / video).with_suffix(".pkl"),
        dataset_root / "hamer" / hamer_video,
    )


def validate_rows(dataset_root, rows, split):
    basenames = [Path(row["video"]).name for row in rows]
    duplicates = sorted(name for name, count in Counter(basenames).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{split} contains duplicate basenames, which the released extractor "
            f"cannot distinguish: {duplicates[:5]}"
        )

    missing = []
    for row in rows:
        if not row.get("text"):
            raise ValueError(f"Missing text in {split} row: {row}")
        for path in condition_paths(dataset_root, row):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{split} is missing {len(missing)} required inputs. First paths:\n{preview}"
        )


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    train_rows = load_metadata(args.train_meta)
    dev_rows = load_metadata(args.dev_meta)
    validate_rows(args.dataset_root, train_rows, "train")
    validate_rows(args.dataset_root, dev_rows, "dev")

    train_names = {Path(row["video"]).name for row in train_rows}
    dev_names = {Path(row["video"]).name for row in dev_rows}
    overlap = sorted(train_names & dev_names)
    if overlap:
        raise ValueError(f"Train/dev basename overlap: {overlap[:5]}")

    shard_zero = train_rows[::2]
    shard_one = train_rows[1::2]
    write_json(args.output_dir / "preflight.json", train_rows[:1])
    write_json(args.output_dir / "train_shard0.json", shard_zero)
    write_json(args.output_dir / "train_shard1.json", shard_one)
    write_json(args.output_dir / "dev.json", dev_rows)

    print(
        json.dumps(
            {
                "train": len(train_rows),
                "train_shard0": len(shard_zero),
                "train_shard1": len(shard_one),
                "dev": len(dev_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
