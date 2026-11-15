#!/bin/bash
set -euo pipefail

FINAL=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/replication/final_phoenix

FSQ_JOB_ID=$(sbatch --parsable "$FINAL/train_fsq.slurm")
FSQ_JOB_ID=${FSQ_JOB_ID%%;*}
FSQ_EVAL_JOB_ID=$(sbatch --parsable --dependency="afterok:$FSQ_JOB_ID" \
  --export="ALL,FSQ_JOB_ID=$FSQ_JOB_ID" "$FINAL/evaluate_fsq.slurm")
FSQ_EVAL_JOB_ID=${FSQ_EVAL_JOB_ID%%;*}
GPT_JOB_ID=$(sbatch --parsable --dependency="afterok:$FSQ_EVAL_JOB_ID" \
  --export="ALL,FSQ_JOB_ID=$FSQ_JOB_ID,FSQ_EVAL_JOB_ID=$FSQ_EVAL_JOB_ID" \
  "$FINAL/train_gpt.slurm")
GPT_JOB_ID=${GPT_JOB_ID%%;*}
EVAL_JOB_ID=$(sbatch --parsable --dependency="afterok:$GPT_JOB_ID" \
  --export="ALL,FSQ_JOB_ID=$FSQ_JOB_ID,GPT_JOB_ID=$GPT_JOB_ID" \
  "$FINAL/evaluate_gpt.slurm")
EVAL_JOB_ID=${EVAL_JOB_ID%%;*}

printf 'fsq_train=%s\nfsq_eval=%s\ngpt_train=%s\ngpt_eval=%s\n' \
  "$FSQ_JOB_ID" "$FSQ_EVAL_JOB_ID" "$GPT_JOB_ID" "$EVAL_JOB_ID" \
  | tee "$FINAL/submitted_jobs.txt"

