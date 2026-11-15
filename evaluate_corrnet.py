#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated sign videos with a frozen CorrNet+ model."
    )
    parser.add_argument("--eval-spec", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--corrnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--gloss-dict",
        type=Path,
        help="Gloss dictionary NPY; defaults to the Isharah1000 dictionary.",
    )
    parser.add_argument(
        "--evaluator-name",
        default="CorrNet+",
        help="Human-readable evaluator identity stored in the result summary.",
    )
    parser.add_argument("--generator-name", default="unspecified")
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument(
        "--allow-missing", action="store_true", help="Score available videos only."
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_video(path: Path, input_size: int, max_frames: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode generated video: {path}")
    if len(frames) > max_frames:
        keep = np.linspace(0, len(frames) - 1, max_frames, dtype=np.int64)
        frames = [frames[index] for index in keep]

    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    height, width = video.shape[-2:]
    scale = input_size / min(height, width)
    resized_height = max(input_size, round(height * scale))
    resized_width = max(input_size, round(width * scale))
    video = F.interpolate(
        video,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - input_size) // 2
    left = (resized_width - input_size) // 2
    video = video[:, :, top : top + input_size, left : left + input_size]
    return video.div(127.5).sub(1.0).unsqueeze(0)


def pad_for_temporal_conv(video, kernel_sizes):
    """Match CorrNet+'s evaluation collate padding for a singleton video."""
    left_pad = 0
    last_stride = 1
    total_stride = 1
    for layer in kernel_sizes:
        if layer[0] == "K":
            left_pad = left_pad * last_stride + (int(layer[1:]) - 1) // 2
        elif layer[0] == "P":
            last_stride = int(layer[1:])
            total_stride *= last_stride
    frames = video.shape[1]
    padded_frames = int(np.ceil(frames / total_stride)) * total_stride + 2 * left_pad
    minimum_frames = 2
    for layer in reversed(kernel_sizes):
        if layer[0] == "K":
            minimum_frames += int(layer[1:]) - 1
        elif layer[0] == "P":
            minimum_frames *= int(layer[1:])
    padded_frames = max(padded_frames, minimum_frames)
    right_pad = padded_frames - frames - left_pad
    padded = torch.cat(
        [
            video[:, :1].expand(-1, left_pad, -1, -1, -1),
            video,
            video[:, -1:].expand(-1, right_pad, -1, -1, -1),
        ],
        dim=1,
    )
    return padded, padded_frames


def edit_alignment(reference, hypothesis):
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    ops = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], ops[i][0] = i, "D"
    for j in range(1, cols):
        costs[0][j], ops[0][j] = j, "I"
    priority = {"C": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            candidates = []
            if reference[i - 1] == hypothesis[j - 1]:
                candidates.append((costs[i - 1][j - 1], "C"))
            else:
                candidates.append((costs[i - 1][j - 1] + 1, "S"))
            candidates.append((costs[i - 1][j] + 1, "D"))
            candidates.append((costs[i][j - 1] + 1, "I"))
            costs[i][j], ops[i][j] = min(
                candidates, key=lambda item: (item[0], priority[item[1]])
            )

    i, j = len(reference), len(hypothesis)
    counts = Counter()
    while i or j:
        op = ops[i][j]
        counts[op] += 1
        if op in ("C", "S"):
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        else:
            j -= 1
    return counts


def corpus_bleu(references, hypotheses, max_order):
    """Compute cumulative corpus BLEU-N with one tokenized reference per sample."""
    matches = [0] * max_order
    possible = [0] * max_order
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, max_order + 1):
            reference_ngrams = Counter(
                tuple(reference[i : i + order])
                for i in range(len(reference) - order + 1)
            )
            hypothesis_ngrams = Counter(
                tuple(hypothesis[i : i + order])
                for i in range(len(hypothesis) - order + 1)
            )
            matches[order - 1] += sum(
                (reference_ngrams & hypothesis_ngrams).values()
            )
            possible[order - 1] += max(len(hypothesis) - order + 1, 0)

    if hypothesis_length == 0:
        return 0.0
    precisions = [
        match / total if total else 0.0
        for match, total in zip(matches, possible)
    ]
    if any(precision == 0.0 for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


def unwrap_state_dict(checkpoint):
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint.items()
    }


def load_corrnet(args, gloss_dict):
    sys.path.insert(0, str(args.corrnet_root.resolve()))
    from slr_network import SLRModel

    model = SLRModel(
        num_classes=len(gloss_dict) + 1,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=1,
        gloss_dict=gloss_dict,
        loss_weights={
            "SeqCTC": 1.0,
            "ConvCTC": 1.0,
            "Dist": 25.0,
            "Cu": 0.0005,
            "Cp": 0.0005,
        },
        weight_norm=True,
        share_classifier=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(unwrap_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"CorrNet checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    model.requires_grad_(False).eval().to(args.device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("R_eval must be fully frozen.")
    return model


def main():
    args = parse_args()
    spec = json.loads(args.eval_spec.read_text(encoding="utf-8"))
    gloss_path = args.gloss_dict or (
        args.corrnet_root / "preprocess/isharah1000/gloss_dict.npy"
    )
    gloss_dict = np.load(gloss_path, allow_pickle=True).item()
    model = load_corrnet(args, gloss_dict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    totals = Counter()
    missing_videos = []
    for sample in spec["samples"]:
        video_path = args.video_dir / sample["video"]
        if not video_path.is_file():
            missing_videos.append(sample["video"])
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing generated video: {video_path}")

        video = load_video(video_path, args.input_size, args.max_frames)
        original_frames = video.shape[1]
        video, padded_frames = pad_for_temporal_conv(
            video, model.conv1d.kernel_size
        )
        video = video.to(args.device)
        video_length = torch.tensor(
            [padded_frames], dtype=torch.long, device=args.device
        )
        with torch.inference_mode():
            output = model(video, video_length)
        hypothesis = [word for word, _ in output["recognized_sents"][0]]
        reference_text = sample.get("reference_gloss") or sample["text"]
        reference = reference_text.split()
        unknown = [word for word in reference if word not in gloss_dict]
        if unknown:
            raise ValueError(
                f"Reference for {sample['id']} contains unknown glosses: {unknown}"
            )
        counts = edit_alignment(reference, hypothesis)
        totals.update(counts)
        errors = counts["S"] + counts["D"] + counts["I"]
        records.append(
            {
                "id": sample["id"],
                "video": sample["video"],
                "reference": reference,
                "hypothesis": hypothesis,
                "prompt_text": sample.get("prompt_text", sample["text"]),
                "reference_text": " ".join(reference),
                "hypothesis_text": " ".join(hypothesis),
                "substitutions": counts["S"],
                "deletions": counts["D"],
                "insertions": counts["I"],
                "correct": counts["C"],
                "errors": errors,
                "exact": reference == hypothesis,
                "num_video_frames": int(original_frames),
                "num_padded_frames": int(padded_frames),
            }
        )

    reference_words = totals["C"] + totals["S"] + totals["D"]
    total_errors = totals["S"] + totals["D"] + totals["I"]
    references = [record["reference"] for record in records]
    hypotheses = [record["hypothesis"] for record in records]
    bleu = {
        order: corpus_bleu(references, hypotheses, order) for order in range(1, 5)
    }
    summary = {
        "protocol": spec["protocol"],
        "provisional": True,
        "evaluator": args.evaluator_name,
        "evaluator_checkpoint": str(args.checkpoint.resolve()),
        "evaluator_sha256": sha256(args.checkpoint),
        "generator": args.generator_name,
        "generator_checkpoint": (
            str(args.generator_checkpoint.resolve())
            if args.generator_checkpoint
            else None
        ),
        "generator_sha256": (
            sha256(args.generator_checkpoint) if args.generator_checkpoint else None
        ),
        "eval_spec": str(args.eval_spec.resolve()),
        "eval_spec_sha256": sha256(args.eval_spec),
        "requested_videos": len(spec["samples"]),
        "evaluated_videos": len(records),
        "coverage": len(records) / len(spec["samples"]),
        "missing_videos": missing_videos,
        "reference_words": reference_words,
        "substitutions": totals["S"],
        "deletions": totals["D"],
        "insertions": totals["I"],
        "correct": totals["C"],
        "wer_percent": 100.0 * total_errors / max(reference_words, 1),
        "bleu_implementation": "Papineni cumulative corpus BLEU, no smoothing",
        "bleu_1": bleu[1],
        "bleu_2": bleu[2],
        "bleu_3": bleu[3],
        "bleu_4": bleu[4],
        "bleu_1_percent": 100.0 * bleu[1],
        "bleu_2_percent": 100.0 * bleu[2],
        "bleu_3_percent": 100.0 * bleu[3],
        "bleu_4_percent": 100.0 * bleu[4],
        "exact_sentence_accuracy_percent": (
            100.0 * sum(record["exact"] for record in records) / max(len(records), 1)
        ),
    }
    with (args.output_dir / "hypotheses.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
