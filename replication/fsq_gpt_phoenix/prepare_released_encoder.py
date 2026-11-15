#!/usr/bin/env python3
"""Remove only the released FSQ tensors from a SignViP condition checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", mmap=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {args.input}")

    vq_keys = [name for name in state if name.startswith("vq.")]
    retained = {name: value for name, value in state.items() if name not in vq_keys}
    motion_keys = [name for name in retained if name.startswith("motion_modules.")]
    if len(vq_keys) != 270:
        raise RuntimeError(f"Expected 270 released vq.* tensors, found {len(vq_keys)}")
    if len(motion_keys) != 140:
        raise RuntimeError(
            f"Expected 140 retained temporal-motion tensors, found {len(motion_keys)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(retained, args.output)
    report = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "source_tensor_count": len(state),
        "removed_vq_tensor_count": len(vq_keys),
        "retained_tensor_count": len(retained),
        "retained_motion_tensor_count": len(motion_keys),
        "output": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
