#!/bin/bash
#SBATCH --job-name=sv_final_gpt50k
#SBATCH --output=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_gpt50k_out_%j.txt
#SBATCH --error=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_gpt50k_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=A100
#SBATCH --gres=gpu:2

set -euo pipefail
source ~/.bashrc
: "${FSQ_JOB_ID:?FSQ_JOB_ID must identify final FSQ training}"
: "${FSQ_EVAL_JOB_ID:?FSQ_EVAL_JOB_ID must identify successful oracle evaluation}"

ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
FINAL=$SIGNVIP/replication/final_phoenix
HELPERS=$SIGNVIP/replication/fsq_gpt_phoenix
GPT_HELPERS=$SIGNVIP/replication/gpt_phoenix
DATA=$ROOT/signvip_dit/data/phoenix2014t
SOURCE_SHA=54a28b326cd969bb03302ba83a0072bfff84a48b
FSQ_RUN=$FINAL/runs/fsq-job-$FSQ_JOB_ID
FSQ_EVAL_RUN=$FSQ_RUN/oracle-job-$FSQ_EVAL_JOB_ID
RUN=$FINAL/runs/gpt-job-$SLURM_JOB_ID
TOKEN_ROOT=$RUN/tokens
JOB_TMP=${SLURM_TMPDIR:-/tmp}/sv-final-gpt-$SLURM_JOB_ID
CLEAN_SRC=$JOB_TMP/upstream
CONFIG=$RUN/train_gpt.yaml

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=$RUN/wandb
export TRITON_CACHE_DIR=$JOB_TMP/triton
export TORCH_EXTENSIONS_DIR=$JOB_TMP/torch_extensions
export MPLCONFIGDIR=$JOB_TMP/matplotlib
export TMPDIR=$JOB_TMP/tmp

mkdir -p "$SIGNVIP/logs" "$RUN/generated" "$RUN/source_snapshot" \
    "$CLEAN_SRC" "$WANDB_DIR" "$TOKEN_ROOT/train_processed_videos" \
    "$TOKEN_ROOT/dev_processed_videos" \
    "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$TMPDIR"
test -f "$FSQ_RUN/complete.txt"
test -f "$FSQ_EVAL_RUN/complete.txt"
FSQ_CHECKPOINT=$(<"$FSQ_RUN/fsq_checkpoint.txt")
test -f "$FSQ_CHECKPOINT"
git -C "$SIGNVIP" archive "$SOURCE_SHA" | tar -x -C "$CLEAN_SRC"

cp "$SIGNVIP/models/multihead_t2vqpgpt.py" \
  "$CLEAN_SRC/models/multihead_t2vqpgpt.py"
cp "$SIGNVIP/train_multihead_t2vqpgpt.py" \
  "$CLEAN_SRC/train_multihead_t2vqpgpt.py"
cp "$SIGNVIP/decoded_fsq_metrics.py" "$CLEAN_SRC/decoded_fsq_metrics.py"
cp "$SIGNVIP/test_gpt_attention.py" "$CLEAN_SRC/test_gpt_attention.py"
patch -d "$CLEAN_SRC" -p1 --dry-run < "$FINAL/gpt_dual_metrics.patch"
patch -d "$CLEAN_SRC" -p1 < "$FINAL/gpt_dual_metrics.patch"

printf '%s\n' "$SOURCE_SHA" > "$RUN/base_source_commit.txt"
printf '%s\n' "$FSQ_CHECKPOINT" > "$RUN/fsq_checkpoint.txt"
cp "$CLEAN_SRC/models/multihead_t2vqpgpt.py" \
  "$CLEAN_SRC/train_multihead_t2vqpgpt.py" "$CLEAN_SRC/decoded_fsq_metrics.py" \
  "$FINAL/gpt_dual_metrics.patch" "$RUN/source_snapshot/"
sha256sum "$RUN/source_snapshot"/* > "$RUN/audited_source.sha256"

conda run --no-capture-output -n signvip python "$GPT_HELPERS/prepare_metadata.py" \
  --dataset-root "$DATA" --train-meta "$DATA/train_meta.json" \
  --dev-meta "$DATA/dev_meta.json" --output-dir "$RUN/generated"

render_extract() {
  local name=$1
  local metadata=$2
  conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
    --kind extract --input "$HELPERS/extract_template.yaml" \
    --output "$RUN/$name.yaml" --vq-checkpoint "$FSQ_CHECKPOINT" \
    --metadata "$metadata"
}
render_extract extract_train0 "$RUN/generated/train_shard0.json"
render_extract extract_train1 "$RUN/generated/train_shard1.json"
render_extract extract_dev "$RUN/generated/dev.json"

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n signvip \
  python "$CLEAN_SRC/get_compress_vq_pose_latent.py" \
  --config "$RUN/extract_train0.yaml" \
  --output_dir "$TOKEN_ROOT/train_processed_videos" \
  > "$RUN/extract_train0.log" 2>&1 &
PID_ZERO=$!
(
  export CUDA_VISIBLE_DEVICES=1
  conda run --no-capture-output -n signvip \
    python "$CLEAN_SRC/get_compress_vq_pose_latent.py" \
    --config "$RUN/extract_train1.yaml" \
    --output_dir "$TOKEN_ROOT/train_processed_videos"
  conda run --no-capture-output -n signvip \
    python "$CLEAN_SRC/get_compress_vq_pose_latent.py" \
    --config "$RUN/extract_dev.yaml" \
    --output_dir "$TOKEN_ROOT/dev_processed_videos"
) > "$RUN/extract_train1_dev.log" 2>&1 &
PID_ONE=$!
wait "$PID_ZERO"
wait "$PID_ONE"

conda run --no-capture-output -n signvip python "$GPT_HELPERS/verify_tokens.py" \
  --token-root "$TOKEN_ROOT" --meta "$DATA/train_meta.json" \
  --meta "$DATA/dev_meta.json" \
  --pose-size 12 --codebook-size 625 --report "$RUN/token_report.json"

conda run --no-capture-output -n signvip python "$FINAL/render_gpt_config.py" \
  --input "$FINAL/train_gpt.yaml" --output "$CONFIG" \
  --token-root "$TOKEN_ROOT" --fsq-checkpoint "$FSQ_CHECKPOINT"

cd "$CLEAN_SRC"
conda run --no-capture-output -n signvip python test_gpt_attention.py
conda run --no-capture-output -n signvip accelerate launch \
  --config_file "$CLEAN_SRC/accelerate_config_bf16.yaml" \
  --num_machines 1 --num_processes 2 --gpu_ids "0,1" --mixed_precision bf16 \
  "$CLEAN_SRC/train_multihead_t2vqpgpt.py" \
  --config "$CONFIG" --workspace_dir "$RUN/workspace" --report_to wandb

test "$(wc -l < "$RUN/workspace/validation_history.jsonl")" -eq 25
for step in $(seq 2000 2000 50000); do
  test -f "$RUN/workspace/model-$step/t2pgpt/model.bin"
done
printf '%s\n' "$RUN/workspace" > "$RUN/workspace.txt"
printf '%s\n' "$RUN" > "$FINAL/latest_gpt_run.txt"
touch "$RUN/complete.txt"
echo "Final GPT training completed: $RUN"
