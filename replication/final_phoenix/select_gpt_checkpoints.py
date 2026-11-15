#!/usr/bin/env python3
"""Select the two requested GPT checkpoints plus the final checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.gpt_run / "workspace"
    history = [
        json.loads(line)
        for line in (workspace / "validation_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(history) != 25:
        raise RuntimeError(f"Expected 25 validation records, found {len(history)}")

    choices = {
        "best_raw_dtw": min(history, key=lambda row: row["raw_token_dtw_score"]),
        "best_decoded_dtw": min(history, key=lambda row: row["decoded_dtw_score"]),
        "final": next(row for row in history if row["step"] == 50000),
    }
    selected = {}
    for name, metrics in choices.items():
        checkpoint = workspace / f"model-{metrics['step']}" / "t2pgpt/model.bin"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        selected[name] = {
            "step": metrics["step"],
            "checkpoint": str(checkpoint.resolve()),
            "validation": metrics,
        }

    report = {
        "raw_dtw_protocol": (
            "released-code categorical DTW over token IDs, prediction truncated to "
            "ground-truth flattened-token length"
        ),
        "decoded_dtw_protocol": (
            "exact temporal DTW over pooled continuous FSQ-decoder features; "
            "prediction and target are decoded from their categorical FSQ tokens"
        ),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

