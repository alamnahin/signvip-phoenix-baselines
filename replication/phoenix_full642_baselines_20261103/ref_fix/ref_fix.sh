#!/bin/bash
# Generate the 3 PHOENIX-2014T test clips whose processed source mp4 has an undecodable first frame
# (h264 "Invalid NAL unit size"), so combined_t2s_eval.py skipped them for lack of a reference frame.
# Reference frame here = original PHOENIX-2014-T first frame (features/fullFrame-210x260px/.../images0001.png);
# on a valid clip it matches the processed video's first frame at 44.5 dB PSNR, same 210x260 size.
# Same models, configs, eval_spec seeds, 50 steps, CFG 3.5 as the main run.
# Launched as an overlapping step inside job 39714's allocation (srun --jobid=39714 --overlap): no new SLURM request.
set -uo pipefail
ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio
SIGNVIP=$ROOT/signvip
RUN=$SIGNVIP/replication/phoenix_full642_baselines_20261103
FIX=$RUN/ref_fix
CONDA=/opt/anaconda3/bin/conda
CLIPS="11January_2011_Tuesday_tagesschau-8017 29March_2011_Tuesday_tagesschau-8108 11September_2009_Friday_tagesschau-8780"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/SLURM/home/slurm_s202281480/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub TORCH_HOME=/SLURM/home/slurm_s202281480/.cache/torch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$FIX/matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "[$(date -Is)] start on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for m in released retrained; do
  mkdir -p "$FIX/$m/videos"
  cd "$SIGNVIP" || exit 1
  echo "[$(date -Is)] generate $m"
  $CONDA run --no-capture-output -n signvip python "$SIGNVIP/combined_t2s_eval.py" \
    --t2s_config "$FIX/${m}_t2s.yaml" --vq_config "$RUN/${m}_vq.yaml" \
    --video_base_path "$FIX/frames" \
    --output_dir "$FIX/$m/videos" --eval_spec "$RUN/eval_spec.json" \
    --num_inference_steps 50 --guidance_scale 3.5 \
    > "$FIX/$m/generation.log" 2>&1 || echo "$m generation exited non-zero" >&2
  for c in $CLIPS; do
    if [ -f "$FIX/$m/videos/$c.mp4" ] && [ ! -e "$RUN/$m/videos/$c.mp4" ]; then
      cp "$FIX/$m/videos/$c.mp4" "$RUN/$m/videos/.tmp_$c" && mv "$RUN/$m/videos/.tmp_$c" "$RUN/$m/videos/$c.mp4"
    fi
  done
  echo "[$(date -Is)] $m now has $(find "$RUN/$m/videos" -maxdepth 1 -name '*.mp4' | wc -l) mp4"
done
echo "[$(date -Is)] done"
