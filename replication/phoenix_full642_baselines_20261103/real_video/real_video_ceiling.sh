#!/bin/bash
# Real-video ceiling: the real PHOENIX-2014T test videos through the same CorrNet+ evaluator,
# same eval_spec (642 clips) and same ROUGE step as the two SignViP baselines.
# This is the step that ran inside job 40301 on 2026-11-05 (21:09-21:19), copied out of
# ssl_motion/experiments/20261105_diagnostics/diagnostics.slurm so it can be rerun on its own.
# Outputs from that run are in ./corrnet/ (summary.json, hypotheses.jsonl).
set -uo pipefail
ROOT=${ROOT:-/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio}
SIGNVIP=$ROOT/signvip
CORRNET=$ROOT/CORRNET/CorrNet_Plus_CSLR
CONDA=${CONDA:-/opt/anaconda3/bin/conda}
PHX_SPEC=$SIGNVIP/replication/phoenix_full642_baselines_20261103/eval_spec.json
PHX_VIDEOS=$ROOT/signvip_dit/data/phoenix2014t/rgb/test_processed_videos
ADD_ROUGE=$SIGNVIP/replication/final_phoenix/add_rouge.py
OUT=${OUT:-$SIGNVIP/replication/phoenix_full642_baselines_20261103/real_video}

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT/corrnet"
cd "$ROOT" || exit 1
$CONDA run --no-capture-output -n corrnet python "$SIGNVIP/evaluate_corrnet.py" \
  --eval-spec "$PHX_SPEC" --video-dir "$PHX_VIDEOS" \
  --corrnet-root "$CORRNET" --checkpoint "$CORRNET/checkpoints/phoenix2014t.pt" \
  --gloss-dict "$CORRNET/preprocess/phoenix2014-T/gloss_dict.npy" \
  --evaluator-name "CorrNet+ Phoenix2014T released checkpoint" \
  --generator-name "PHOENIX-2014T real test video (processed)" \
  --output-dir "$OUT/corrnet" --device cuda:0 --input-size 224 \
  > "$OUT/corrnet.log" 2>&1 || { echo "CorrNet+ failed (see $OUT/corrnet.log)" >&2; exit 1; }
$CONDA run --no-capture-output -n signvip python "$ADD_ROUGE" --corrnet-dir "$OUT/corrnet" \
  > "$OUT/rouge.log" 2>&1 || { echo "ROUGE failed (see $OUT/rouge.log)" >&2; exit 1; }
echo "done: $OUT/corrnet/summary.json"
