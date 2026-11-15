#!/bin/bash
#SBATCH --job-name=sv_final_gpt_eval
#SBATCH --output=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_gpt_eval_out_%j.txt
#SBATCH --error=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_gpt_eval_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=A100
#SBATCH --gres=gpu:2

set -euo pipefail
source ~/.bashrc
: "${FSQ_JOB_ID:?FSQ_JOB_ID must identify final FSQ training}"
: "${GPT_JOB_ID:?GPT_JOB_ID must identify final GPT training}"

ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
FINAL=$SIGNVIP/replication/final_phoenix
FSQ_HELPERS=$SIGNVIP/replication/fsq_gpt_phoenix
GPT_HELPERS=$SIGNVIP/replication/gpt_phoenix
DATA=$ROOT/signvip_dit/data/phoenix2014t
CORRNET=$ROOT/CORRNET/CorrNet_Plus_CSLR
CORRNET_CKPT=$CORRNET/checkpoints/phoenix2014t.pt
SPEC=$SIGNVIP/eval_specs/phoenix2014t_test_first100.json
FSQ_RUN=$FINAL/runs/fsq-job-$FSQ_JOB_ID
GPT_RUN=$FINAL/runs/gpt-job-$GPT_JOB_ID
RUN=$FINAL/results/job-$SLURM_JOB_ID
JOB_TMP=${SLURM_TMPDIR:-/tmp}/sv-final-gpt-eval-$SLURM_JOB_ID

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/SLURM/home/slurm_s202281480/.cache/torch
export MPLCONFIGDIR=$JOB_TMP/matplotlib
export TMPDIR=$JOB_TMP/tmp

mkdir -p "$SIGNVIP/logs" "$RUN" "$MPLCONFIGDIR" "$TMPDIR"
test -f "$FSQ_RUN/complete.txt"
test -f "$GPT_RUN/complete.txt"
test -f "$SPEC"
FSQ_CHECKPOINT=$(<"$FSQ_RUN/fsq_checkpoint.txt")
test -f "$FSQ_CHECKPOINT"

conda run --no-capture-output -n signvip python "$FINAL/select_gpt_checkpoints.py" \
  --gpt-run "$GPT_RUN" --output "$RUN/selected_checkpoints.json"
conda run --no-capture-output -n signvip python "$FSQ_HELPERS/make_runtime_config.py" \
  --kind vq-eval --input "$SIGNVIP/configs/eval/vq_test_phoenix100.yaml" \
  --output "$RUN/vq.yaml" --vq-checkpoint "$FSQ_CHECKPOINT"

checkpoint_for() {
  conda run --no-capture-output -n signvip python -c \
    "import json; print(json.load(open('$RUN/selected_checkpoints.json'))['selected']['$1']['checkpoint'])"
}

generate() {
  local gpu=$1
  local name=$2
  local checkpoint=$3
  local output=$RUN/$name
  mkdir -p "$output/videos"
  conda run --no-capture-output -n signvip python "$GPT_HELPERS/make_runtime_config.py" \
    --kind corrected-eval --input "$SIGNVIP/configs/eval/t2s_test_phoenix100.yaml" \
    --output "$output/t2s.yaml" --checkpoint "$checkpoint" --attention global
  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n signvip \
    python "$SIGNVIP/combined_t2s_eval.py" \
    --t2s_config "$output/t2s.yaml" --vq_config "$RUN/vq.yaml" \
    --video_base_path "$DATA/rgb/test_processed_videos" \
    --output_dir "$output/videos" --eval_spec "$SPEC" \
    --num_inference_steps 50 --guidance_scale 3.5 \
    > "$output/generation.log" 2>&1
}

RAW_CHECKPOINT=$(checkpoint_for best_raw_dtw)
DECODED_CHECKPOINT=$(checkpoint_for best_decoded_dtw)
FINAL_CHECKPOINT=$(checkpoint_for final)
generate 0 best_raw_dtw "$RAW_CHECKPOINT" & PID_RAW=$!
generate 1 best_decoded_dtw "$DECODED_CHECKPOINT" & PID_DECODED=$!
wait "$PID_RAW"
wait "$PID_DECODED"
generate 0 final "$FINAL_CHECKPOINT"

evaluate_one() {
  local name=$1
  local checkpoint=$2
  local output=$RUN/$name
  mkdir -p "$output/corrnet"
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n corrnet \
    python "$SIGNVIP/evaluate_corrnet.py" \
    --eval-spec "$SPEC" --video-dir "$output/videos" \
    --corrnet-root "$CORRNET" --checkpoint "$CORRNET_CKPT" \
    --gloss-dict "$CORRNET/preprocess/phoenix2014-T/gloss_dict.npy" \
    --evaluator-name "CorrNet+ Phoenix2014T released checkpoint" \
    --generator-name "Final SignViP GPT $name" \
    --generator-checkpoint "$checkpoint" --output-dir "$output/corrnet" \
    --device cuda:0 --input-size 224 --allow-missing \
    > "$output/corrnet.log" 2>&1
  conda run --no-capture-output -n signvip python "$FINAL/add_rouge.py" \
    --corrnet-dir "$output/corrnet" > "$output/rouge.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n signvip \
    python "$FINAL/evaluate_visual_metrics.py" \
    --eval-spec "$SPEC" --video-dir "$output/videos" \
    --output "$output/visual_metrics.json" --device cuda:0 \
    --num-frames 16 --batch-size 16 > "$output/visual_metrics.log" 2>&1
}

evaluate_one best_raw_dtw "$RAW_CHECKPOINT"
evaluate_one best_decoded_dtw "$DECODED_CHECKPOINT"
evaluate_one final "$FINAL_CHECKPOINT"

conda run --no-capture-output -n signvip python "$FINAL/write_final_report.py" \
  --run "$RUN" > "$RUN/report.log" 2>&1
sha256sum "$FINAL"/*.py "$FINAL"/*.yaml "$FINAL"/*.patch \
  > "$RUN/pipeline_sources.sha256"
printf '%s\n' "$RUN" > "$FINAL/latest_results_run.txt"
touch "$RUN/complete.txt"
echo "Final SignViP evaluation completed: $RUN"
