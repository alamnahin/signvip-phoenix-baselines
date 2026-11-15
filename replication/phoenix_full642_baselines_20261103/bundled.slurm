#!/bin/bash
#SBATCH --job-name=sv_phx_full_baselines
#SBATCH --output=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/replication/phoenix_full642_baselines_20261103/bundled-%j.out
#SBATCH --error=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/replication/phoenix_full642_baselines_20261103/bundled-%j.err
#SBATCH --partition=A6000,A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# One allocation, one GPU (single-GPU A6000 nodes are the ones that free up):
# generation (released, then retrained SignViP) -> CorrNet+ back-translation -> video-quality metrics -> summary.
# No retraining. Resumable: combined_t2s_eval.py skips videos that already exist.
# Timing from the 100-clip runs: retrained ~76 min/100 clips, released ~20 min/100 clips.
set -uo pipefail
source ~/.bashrc
ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
FINAL=$SIGNVIP/replication/final_phoenix
RUN=$SIGNVIP/replication/phoenix_full642_baselines_20261103
DATA=$ROOT/signvip_dit/data/phoenix2014t
CORRNET=$ROOT/CORRNET/CorrNet_Plus_CSLR
CONDA=/opt/anaconda3/bin/conda

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/SLURM/home/slurm_s202281480/.cache/torch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=$RUN/matplotlib

for f in eval_spec.json metadata.json released_t2s.yaml released_vq.yaml retrained_t2s.yaml retrained_vq.yaml evaluate_visual_metrics.py summarize.py; do
  test -f "$RUN/$f" || { echo "missing $RUN/$f" >&2; exit 1; }
done
mkdir -p "$MPLCONFIGDIR"
date -Is > "$RUN/started_$SLURM_JOB_ID.txt"
nvidia-smi > "$RUN/nvidia-smi_$SLURM_JOB_ID.txt" 2>&1 || true

generate() {
  local name=$1
  mkdir -p "$RUN/$name/videos"
  cd "$SIGNVIP" || return 1
  echo "[$(date -Is)] generate $name"
  $CONDA run --no-capture-output -n signvip python "$SIGNVIP/combined_t2s_eval.py" \
    --t2s_config "$RUN/${name}_t2s.yaml" --vq_config "$RUN/${name}_vq.yaml" \
    --video_base_path "$DATA/rgb/test_processed_videos" \
    --output_dir "$RUN/$name/videos" --eval_spec "$RUN/eval_spec.json" \
    --num_inference_steps 50 --guidance_scale 3.5 \
    >> "$RUN/$name/generation.log" 2>&1
}

score() {
  local name=$1 label=$2 ckpt=$3
  local out=$RUN/$name count
  count=$(find "$out/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  echo "[$(date -Is)] score $name ($count videos)" | tee "$out/video_count.txt"
  [ "$count" -eq 642 ] || { echo "$name: expected 642 videos, found $count" >&2; return 1; }
  mkdir -p "$out/corrnet"
  cd "$ROOT" || return 1
  $CONDA run --no-capture-output -n corrnet python "$SIGNVIP/evaluate_corrnet.py" \
    --eval-spec "$RUN/eval_spec.json" --video-dir "$out/videos" \
    --corrnet-root "$CORRNET" --checkpoint "$CORRNET/checkpoints/phoenix2014t.pt" \
    --gloss-dict "$CORRNET/preprocess/phoenix2014-T/gloss_dict.npy" \
    --evaluator-name "CorrNet+ Phoenix2014T released checkpoint" \
    --generator-name "$label" --generator-checkpoint "$ckpt" \
    --output-dir "$out/corrnet" --device cuda:0 --input-size 224 \
    > "$out/corrnet.log" 2>&1 || return 1
  $CONDA run --no-capture-output -n signvip python "$FINAL/add_rouge.py" --corrnet-dir "$out/corrnet" \
    > "$out/rouge.log" 2>&1 || return 1
  $CONDA run --no-capture-output -n signvip python "$RUN/evaluate_visual_metrics.py" \
    --eval-spec "$RUN/eval_spec.json" --video-dir "$out/videos" \
    --output "$out/visual_metrics.json" --device cuda:0 --num-frames 16 --batch-size 16 \
    > "$out/visual_metrics.log" 2>&1 || return 1
}

# A failure in one model does not stop the other; the job exits non-zero at the end if anything failed.
STATUS=0
generate released  || { echo "released generation failed (see released/generation.log)" >&2; STATUS=1; }
generate retrained || { echo "retrained generation failed (see retrained/generation.log)" >&2; STATUS=1; }
date -Is > "$RUN/generation_done_$SLURM_JOB_ID.txt"

score released "SignViP released Phoenix2014T checkpoint (full test)" \
  "$SIGNVIP/checkpoints/RWTH/t2gpt.bin" \
  || { echo "released scoring failed (see released/*.log)" >&2; STATUS=1; }
score retrained "SignViP corrected replication final GPT (full test)" \
  "$FINAL/runs/gpt-job-25779/workspace/model-50000/t2pgpt/model.bin" \
  || { echo "retrained scoring failed (see retrained/*.log)" >&2; STATUS=1; }

$CONDA run --no-capture-output -n signvip python "$RUN/summarize.py" > "$RUN/summary.log" 2>&1 \
  || { echo "summary failed (see summary.log)" >&2; STATUS=1; }
date -Is > "$RUN/complete_$SLURM_JOB_ID.txt"
echo "Done (status $STATUS): $RUN/RESULTS.md"
exit $STATUS
