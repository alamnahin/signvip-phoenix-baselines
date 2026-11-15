#!/bin/bash
#SBATCH --job-name=sv_final_fsq_eval
#SBATCH --output=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_fsq_eval_out_%j.txt
#SBATCH --error=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_fsq_eval_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=A100
#SBATCH --gres=gpu:2

set -euo pipefail
source ~/.bashrc
: "${FSQ_JOB_ID:?FSQ_JOB_ID must identify final FSQ training}"

ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
FINAL=$SIGNVIP/replication/final_phoenix
HELPERS=$SIGNVIP/replication/fsq_gpt_phoenix
DATA=$ROOT/signvip_dit/data/phoenix2014t
CORRNET=$ROOT/CORRNET/CorrNet_Plus_CSLR
CORRNET_CKPT=$CORRNET/checkpoints/phoenix2014t.pt
SOURCE_SHA=54a28b326cd969bb03302ba83a0072bfff84a48b
FSQ_RUN=$FINAL/runs/fsq-job-$FSQ_JOB_ID
RUN=$FSQ_RUN/oracle-job-$SLURM_JOB_ID
JOB_TMP=${SLURM_TMPDIR:-/tmp}/sv-final-fsq-eval-$SLURM_JOB_ID
CLEAN_SRC=$JOB_TMP/upstream
BASE_ENCODER=$JOB_TMP/condition_encoder_without_vq.bin
DIAGNOSTIC_CONFIG=$RUN/diagnostic.yaml
META=$RUN/phoenix2014t_dev_first100_meta.json
SPEC=$RUN/phoenix2014t_dev_first100.json

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/SLURM/home/slurm_s202281480/.cache/torch
export MPLCONFIGDIR=$JOB_TMP/matplotlib
export TMPDIR=$JOB_TMP/tmp

mkdir -p "$SIGNVIP/logs" "$RUN" "$CLEAN_SRC" "$MPLCONFIGDIR" "$TMPDIR"
test -f "$FSQ_RUN/complete.txt"
test -f "$CORRNET_CKPT"
git -C "$SIGNVIP" archive "$SOURCE_SHA" | tar -x -C "$CLEAN_SRC"
WORKSPACE=$(<"$FSQ_RUN/workspace.txt")
test -d "$WORKSPACE"

conda run --no-capture-output -n signvip python \
  "$HELPERS/prepare_released_encoder.py" \
  --input "$SIGNVIP/checkpoints/RWTH/condition_encoder.bin" \
  --output "$BASE_ENCODER" --report "$RUN/encoder_provenance.json"
conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
  --kind fsq --input "$FINAL/train_fsq.yaml" \
  --output "$DIAGNOSTIC_CONFIG" --base-encoder "$BASE_ENCODER"

sha256sum "$SIGNVIP/combined_t2s_eval.py" "$SIGNVIP/diagnose_fsq.py" \
  "$SIGNVIP/evaluate_corrnet.py" "$FINAL/add_rouge.py" \
  > "$RUN/evaluator_sources.sha256"

conda run --no-capture-output -n signvip python \
  "$SIGNVIP/prepare_phoenix2014t_eval.py" \
  --metadata "$DATA/dev_meta.json" \
  --video-root "$DATA/rgb/dev_processed_videos" \
  --recognition-info "$CORRNET/preprocess/phoenix2014-T/dev_info.npy" \
  --split dev --limit 100 \
  --subset-metadata "$META" --eval-spec "$SPEC"

evaluate_step() {
  local step=$1
  local gpu=$2
  local step_run=$RUN/step-$step
  local checkpoint=$WORKSPACE/model-$step/condition_encoder/model.bin
  local tokens=$step_run/tokens/dev_processed_videos
  mkdir -p "$tokens" "$step_run/videos" "$step_run/corrnet"
  test -f "$checkpoint"

  conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
    --kind extract --input "$HELPERS/extract_template.yaml" \
    --output "$step_run/extract.yaml" --vq-checkpoint "$checkpoint" \
    --metadata "$META" --rgb-root "$DATA/rgb/dev_processed_videos" \
    --dwpose-root "$DATA/dwpose/dev_processed_videos" \
    --hamer-root "$DATA/hamer/dev_processed_videos"

  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n signvip \
    python "$SIGNVIP/diagnose_fsq.py" \
    --config "$DIAGNOSTIC_CONFIG" --vq-checkpoint "$checkpoint" \
    --metadata "$DATA/dev_meta.json" --output "$step_run/fsq_diagnostics.json" \
    --max-samples 519 --num-frames 32 --sample-rate 1 \
    > "$step_run/diagnostics.log" 2>&1

  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n signvip \
    python "$CLEAN_SRC/get_compress_vq_pose_latent.py" \
    --config "$step_run/extract.yaml" --output_dir "$tokens" \
    > "$step_run/extract.log" 2>&1

  conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
    --kind oracle-eval --input "$SIGNVIP/configs/eval/t2s_test_phoenix100.yaml" \
    --output "$step_run/oracle.yaml" --metadata "$META" \
    --rgb-root "$DATA/rgb/dev_processed_videos" --token-root "$tokens"
  conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
    --kind vq-eval --input "$SIGNVIP/configs/eval/vq_test_phoenix100.yaml" \
    --output "$step_run/vq.yaml" --vq-checkpoint "$checkpoint"

  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n signvip \
    python "$SIGNVIP/combined_t2s_eval.py" \
    --t2s_config "$step_run/oracle.yaml" --vq_config "$step_run/vq.yaml" \
    --video_base_path "$DATA/rgb/dev_processed_videos" \
    --output_dir "$step_run/videos" --eval_spec "$SPEC" \
    --num_inference_steps 50 --guidance_scale 3.5 --use_ground_truth_tokens \
    > "$step_run/generation.log" 2>&1

  local count
  count=$(find "$step_run/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  test "$count" -eq 100
  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n corrnet \
    python "$SIGNVIP/evaluate_corrnet.py" \
    --eval-spec "$SPEC" --video-dir "$step_run/videos" \
    --corrnet-root "$CORRNET" --checkpoint "$CORRNET_CKPT" \
    --gloss-dict "$CORRNET/preprocess/phoenix2014-T/gloss_dict.npy" \
    --evaluator-name "CorrNet+ Phoenix2014T released checkpoint" \
    --generator-name "Final SignViP FSQ oracle step $step" \
    --generator-checkpoint "$checkpoint" --output-dir "$step_run/corrnet" \
    --device cuda:0 --input-size 224 > "$step_run/corrnet.log" 2>&1
  conda run --no-capture-output -n signvip python "$FINAL/add_rouge.py" \
    --corrnet-dir "$step_run/corrnet" > "$step_run/rouge.log" 2>&1
}

evaluate_step 10000 0 & PID_10=$!
evaluate_step 20000 1 & PID_20=$!
wait "$PID_10"
wait "$PID_20"
evaluate_step 30000 0 & PID_30=$!
evaluate_step 40000 1 & PID_40=$!
wait "$PID_30"
wait "$PID_40"
evaluate_step 50000 0

conda run --no-capture-output -n signvip python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
curve = []
for step in (10000, 20000, 30000, 40000, 50000):
    semantic = json.loads((root / f"step-{step}" / "corrnet/summary.json").read_text())
    diagnostic = json.loads((root / f"step-{step}" / "fsq_diagnostics.json").read_text())
    curve.append({
        "step": step,
        "wer_percent": semantic["wer_percent"],
        **{f"bleu_{i}_percent": semantic[f"bleu_{i}_percent"] for i in range(1, 5)},
        "rouge_l_f1_percent": semantic["rouge_l_f1_percent"],
        "used_codes": diagnostic["used_codes"],
        "perplexity": diagnostic["perplexity"],
        "mid_nmse": diagnostic["metrics"]["mid/nmse"]["mean"],
        "mid_cosine": diagnostic["metrics"]["mid/cosine"]["mean"],
        "temporal_delta_nmse": diagnostic["temporal"]["global_delta_nmse"],
        "temporal_delta_cosine": diagnostic["temporal"]["global_delta_cosine"],
    })
(root / "fsq_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
print(json.dumps(curve, indent=2))
PY

printf '%s\n' "$RUN" > "$FINAL/latest_fsq_eval_run.txt"
touch "$RUN/complete.txt"
echo "Final FSQ oracle evaluation completed: $RUN"
