#!/bin/bash
#SBATCH --job-name=sv_final_fsq50k
#SBATCH --output=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_fsq50k_out_%j.txt
#SBATCH --error=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/logs/sv_final_fsq50k_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=A100
#SBATCH --gres=gpu:2

set -euo pipefail
source ~/.bashrc

ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
FINAL=$SIGNVIP/replication/final_phoenix
HELPERS=$SIGNVIP/replication/fsq_gpt_phoenix
SOURCE_SHA=54a28b326cd969bb03302ba83a0072bfff84a48b
RUN=$FINAL/runs/fsq-job-$SLURM_JOB_ID
JOB_TMP=${SLURM_TMPDIR:-/tmp}/sv-final-fsq-$SLURM_JOB_ID
CLEAN_SRC=$JOB_TMP/upstream
BASE_ENCODER=$JOB_TMP/condition_encoder_without_vq.bin
CONFIG=$RUN/train_fsq.yaml

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=$RUN/wandb
export TRITON_CACHE_DIR=$JOB_TMP/triton
export TORCH_EXTENSIONS_DIR=$JOB_TMP/torch_extensions
export MPLCONFIGDIR=$JOB_TMP/matplotlib
export TMPDIR=$JOB_TMP/tmp

mkdir -p "$SIGNVIP/logs" "$RUN" "$CLEAN_SRC" "$WANDB_DIR" \
  "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$TMPDIR"
test "$(git -C "$SIGNVIP" rev-parse "$SOURCE_SHA")" = "$SOURCE_SHA"
git -C "$SIGNVIP" archive "$SOURCE_SHA" | tar -x -C "$CLEAN_SRC"
git -C "$CLEAN_SRC" apply --check "$HELPERS/fsq_fp32_no_cast.patch"
git -C "$CLEAN_SRC" apply "$HELPERS/fsq_fp32_no_cast.patch"
git -C "$CLEAN_SRC" apply --check "$FINAL/fsq_full_validation.patch"
git -C "$CLEAN_SRC" apply "$FINAL/fsq_full_validation.patch"

printf '%s\n' "$SOURCE_SHA" > "$RUN/source_commit.txt"
sha256sum "$HELPERS/fsq_fp32_no_cast.patch" "$FINAL/fsq_full_validation.patch" \
  > "$RUN/source_patches.sha256"
cp "$FINAL/train_fsq.yaml" "$RUN/base_train_fsq.yaml"

conda run --no-capture-output -n signvip python \
  "$HELPERS/prepare_released_encoder.py" \
  --input "$SIGNVIP/checkpoints/RWTH/condition_encoder.bin" \
  --output "$BASE_ENCODER" \
  --report "$RUN/encoder_provenance.json"

conda run --no-capture-output -n signvip python "$HELPERS/make_runtime_config.py" \
  --kind fsq \
  --input "$FINAL/train_fsq.yaml" \
  --output "$CONFIG" \
  --base-encoder "$BASE_ENCODER"

cd "$RUN"
conda run --no-capture-output -n signvip accelerate launch \
  --config_file "$CLEAN_SRC/accelerate_config.yaml" \
  --num_machines 1 --num_processes 2 --gpu_ids "0,1" --mixed_precision no \
  "$CLEAN_SRC/train_compress_vq_multicond.py" \
  --config "$CONFIG" \
  --report_to wandb

WORKSPACE=$(find "$RUN/workspace/train_fsq" -mindepth 1 -maxdepth 1 \
  -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
test -n "$WORKSPACE"
for step in 10000 20000 30000 40000 50000; do
  test -f "$WORKSPACE/model-$step/condition_encoder/model.bin"
done
printf '%s\n' "$WORKSPACE" > "$RUN/workspace.txt"
printf '%s\n' "$WORKSPACE/model-50000/condition_encoder/model.bin" \
  > "$RUN/fsq_checkpoint.txt"
printf '%s\n' "$RUN" > "$FINAL/latest_fsq_run.txt"
touch "$RUN/complete.txt"
echo "Final FSQ training completed: $RUN"
