#!/usr/bin/env python3
"""Add transparent gloss-sequence ROUGE-L scores to a CorrNet+ summary."""

import argparse
import json
from pathlib import Path


def lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrnet-dir", type=Path, required=True)
    args = parser.parse_args()

    hypotheses_path = args.corrnet_dir / "hypotheses.jsonl"
    summary_path = args.corrnet_dir / "summary.json"
    records = [
        json.loads(line)
        for line in hypotheses_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recalls = []
    precisions = []
    f1_scores = []
    for record in records:
        reference = record["reference"]
        hypothesis = record["hypothesis"]
        overlap = lcs_length(reference, hypothesis)
        recall = overlap / max(len(reference), 1)
        precision = overlap / max(len(hypothesis), 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-12)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "rouge_l_implementation": (
                "macro mean of sentence-level token LCS precision/recall/F1; "
                "CorrNet+ gloss output, not paper video-to-spoken-text output"
            ),
            "rouge_l_precision_percent": 100 * sum(precisions) / max(len(records), 1),
            "rouge_l_recall_percent": 100 * sum(recalls) / max(len(records), 1),
            "rouge_l_f1_percent": 100 * sum(f1_scores) / max(len(records), 1),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

