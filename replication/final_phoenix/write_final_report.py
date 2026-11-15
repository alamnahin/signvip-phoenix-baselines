#!/usr/bin/env python3
"""Combine checkpoint, semantic, and visual results into final artifacts."""

import argparse
import json
from pathlib import Path


def value_or_na(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    selections = json.loads((args.run / "selected_checkpoints.json").read_text())
    rows = []
    for name, selected in selections["selected"].items():
        semantic = json.loads((args.run / name / "corrnet/summary.json").read_text())
        visual = json.loads((args.run / name / "visual_metrics.json").read_text())
        metrics = visual["metrics"]
        unavailable = visual["unavailable"]
        rows.append(
            {
                "selection": name,
                "step": selected["step"],
                "checkpoint": selected["checkpoint"],
                "validation": selected["validation"],
                "coverage_percent": 100 * semantic["coverage"],
                "wer_percent": semantic["wer_percent"],
                **{
                    f"bleu_{order}_percent": semantic[f"bleu_{order}_percent"]
                    for order in range(1, 5)
                },
                "rouge_l_f1_percent": semantic["rouge_l_f1_percent"],
                "comet": unavailable["comet"]["value"],
                "fid": metrics["fid_inception"],
                "clip_fid": unavailable["clip_fid"]["value"],
                "fvd": metrics["fvd_i3d"],
                "ids": unavailable["identity_similarity"]["value"],
                "ssim": metrics["ssim"],
                "psnr_db": metrics["psnr_db"],
                "lpips": metrics["lpips_alex"],
                "hand_ssim": unavailable["hand_ssim"]["value"],
            }
        )

    report = {
        "status": "provisional_local_evaluation",
        "warning": (
            "These fixed-subset local scores are not directly comparable to the "
            "full-test SignViP paper tables. CorrNet+ evaluates gloss recognition, "
            "not video-to-spoken-text translation."
        ),
        "selection_protocol": selections,
        "rows": rows,
        "unavailable_metric_reasons": json.loads(
            (args.run / rows[0]["selection"] / "visual_metrics.json").read_text()
        )["unavailable"],
    }
    (args.run / "final_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    headers = [
        "Selection", "Step", "Coverage", "WER", "B1", "B2", "B3", "B4",
        "ROUGE-L", "COMET", "FID", "CLIP-FID", "FVD", "IDS", "SSIM",
        "PSNR", "LPIPS", "Hand SSIM",
    ]
    lines = [
        "# Final Phoenix2014T Results",
        "",
        report["warning"],
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        values = [
            row["selection"], str(row["step"]), value_or_na(row["coverage_percent"]),
            value_or_na(row["wer_percent"]), value_or_na(row["bleu_1_percent"]),
            value_or_na(row["bleu_2_percent"]), value_or_na(row["bleu_3_percent"]),
            value_or_na(row["bleu_4_percent"]), value_or_na(row["rouge_l_f1_percent"]),
            value_or_na(row["comet"]), value_or_na(row["fid"]),
            value_or_na(row["clip_fid"]), value_or_na(row["fvd"]),
            value_or_na(row["ids"]), value_or_na(row["ssim"]),
            value_or_na(row["psnr_db"]), value_or_na(row["lpips"]),
            value_or_na(row["hand_ssim"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "All semantic percentage columns use CorrNet+ gloss hypotheses. FID, FVD,",
            "SSIM, PSNR, and LPIPS use the explicit protocol in each `visual_metrics.json`.",
            "Unavailable metrics are deliberately not replaced with incompatible proxies.",
            "",
        ]
    )
    (args.run / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

