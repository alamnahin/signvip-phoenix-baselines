# SignViP baselines on PHOENIX-2014T

This repo holds the code used to train and score two text-to-sign-video baselines on PHOENIX-2014T:

- **released**: the authors' SignViP Phoenix2014T checkpoint, unchanged;
- **retrained**: our corrected replication. Stage II (FSQ) and Stage III (text-to-token GPT) were retrained, and the released Stage-I generator was kept frozen.

Both are scored on all 642 test clips, next to the real test videos run through the same evaluator.

The repo starts from upstream [umnooob/signvip](https://github.com/umnooob/signvip) at
`54a28b326cd969bb03302ba83a0072bfff84a48b`, with its history kept. Upstream's README is now `README.upstream.md`. Upstream ships no licence, so keep this repo private.

## Results (642 test clips, one run each)

Back-translation is gloss recognition by CorrNet+, not the SignViP paper's video-to-German-text model. These numbers do not go next to the paper's Table 1.

| | WER ↓ | BLEU-1 | BLEU-4 | ROUGE-L | exact % | FVD ↓ | FID ↓ | CLIP-FID ↓ | SSIM | PSNR | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real test video | 21.9 | 80.85 | 59.29 | 81.55 | 26.64 | – | – | – | – | – | – |
| SignViP released | 116.2 | 13.94 | 1.53 | 14.42 | 0.47 | 88.0 | 13.91 | 3.67 | 0.611 | 13.58 | 0.315 |
| SignViP retrained | 115.7 | 14.32 | 2.44 | 15.45 | 0.62 | 95.4 | 14.28 | 3.32 | 0.611 | 13.61 | 0.316 |

The machine-readable results are in `replication/phoenix_full642_baselines_20261103/results.json`, and the per-clip CorrNet+ output is in `*/corrnet/hypotheses.jsonl`. The score report, with BLEU-2 and BLEU-3, gloss error counts and metric notes, is [`phoenix-baselines.md`](phoenix-baselines.md).

## FSQ fixes

Both FSQ fixes are applied to the top-level `train_compress_vq_multicond.py`. Keep `weight_dtype: fp32` when training it, as `configs/vq/vq_multicond_RWTH_compress.yaml` and `replication/final_phoenix/train_fsq.yaml` do.

The patch files stay in the repo because `replication/final_phoenix/train_fsq.slurm`, the job behind the recorded results, extracts upstream commit `54a28b3` to a clean folder and applies them there. That job doesn't use the top-level script, and the top-level script with these fixes has not itself been trained end to end.

## Layout

Paths match the cluster's `signvip/` checkout, so the job scripts run unchanged when this repo sits at `$ROOT/signvip`.

| Path | What it is |
|---|---|
| `*.py`, `models/`, … (modified) | Upstream files as they were on the cluster: Isharah/W&B/checkpoint-loading changes, explicit `attn_type`, decoded-FSQ DTW validation. The GPT training copies `models/multihead_t2vqpgpt.py` and `train_multihead_t2vqpgpt.py` from here. Generation uses `combined_t2s_eval.py`. |
| `evaluate_corrnet.py` | CorrNet+ back-translation: WER, BLEU-1..4, exact match, per-clip hypotheses. |
| `prepare_phoenix2014t_eval.py`, `eval_specs/`, `configs/eval/*phoenix*` | Build the fixed evaluation specs (clip list, reference glosses, per-clip seeds). |
| `decoded_fsq_metrics.py`, `diagnose_fsq.py`, `test_gpt_attention.py` | FSQ/GPT validation helpers used by the training chain. |
| `replication/fsq_gpt_phoenix/` | Helpers shared by the chain. `fsq_fp32_no_cast.patch` fixes the in-place FP16 cast of the trainable FSQ during validation. Also the released-encoder prep, runtime-config renderer and token-extraction template. |
| `replication/gpt_phoenix/` | Token metadata shards, the token verifier and the eval-config renderer. |
| `replication/final_phoenix/` | **Training the retrained baseline:** `submit.sh` chains FSQ train → FSQ oracle eval → GPT train → GPT eval on the first 100 test clips. Also holds configs, patches, `README.md` (fixed protocol) and `SUBMISSION_HISTORY.md`. `runs/` and `results/` keep only the provenance and summary files of jobs 25742/25743/25779/25780. |
| `replication/phoenix_full642_baselines_20261103/` | **Scoring both baselines on the full test set** (job 39714): `prepare.py` → `bundled.slurm` → `summarize.py`. `ref_fix/` regenerates 3 clips whose reference frame failed to load. `real_video/` holds the real-video ceiling. |
| `third_party/corrnet_plus/` | Base commit plus the local patch of the evaluator. Note that it decodes greedily. |

## How the retrained baseline was trained

Both stages start from a clean `git archive 54a28b3` and apply the patches listed in each run's provenance files. `test_gpt_attention.py` runs before GPT training.

1. **Stage II, FSQ** (`train_fsq.slurm`, job 25742):
   - A fresh FSQ autoencoder is distilled from the released condition encoder.
   - 50k steps in FP32, global batch 16 on 2×A100.
   - AdamW, lr 5e-5, no weight decay, cosine schedule, 500 warm-up steps, gradient clip 1.0.
   - Patches: `fsq_fp32_no_cast.patch` and `fsq_full_validation.patch`. Upstream stopped validating after 5 batches; the second patch validates on all of dev.
   - Oracle check (`evaluate_fsq.slurm`, job 25743): ground-truth FSQ tokens → FSQ decoder → released generator.
2. **Stage III, GPT** (`train_gpt.slurm`, job 25779):
   - Train and dev tokens are extracted with the 50k FSQ: 12 tokens per frame, 625 codes, checked by `verify_tokens.py`.
   - Released 8-block architecture with multilingual CLIP ViT-B/32 text features and global attention.
   - 50k steps in BF16, global batch 64 on 2×A100.
   - AdamW8bit, lr 2e-4, weight decay 1e-6, cosine schedule, 200 warm-up steps, 40% input-token corruption (`p_keep 0.6`), gradient clip 1.0.
   - `gpt_dual_metrics.patch` adds teacher-forced loss and accuracy, plus raw-token DTW, to validation.
3. **Checkpoint choice** (`evaluate_gpt.slurm`, job 25780):
   - Three checkpoints were tried on the first 100 test clips.
   - Best raw-DTW and best decoded-DTW were both step 2000. They gave near-static video: WER 100, BLEU-4 0, FVD 939.
   - The final step-50000 checkpoint gave WER 123.0, BLEU-4 2.07, FVD 247.
   - The full-test baseline uses step 50000.

## How both were scored

Text runs through the GPT, then FSQ tokens, then the condition decoder, then the released diffusion generator:

- reference image: the first frame of the source test video;
- DDIM, 50 steps, CFG 3.5;
- seeds fixed per clip in `eval_spec.json`.

Metrics:

- **Recognition:** CorrNet+ on the generated video gives WER, corpus BLEU (no smoothing), sentence-mean ROUGE-L (`add_rouge.py`) and exact match.
- **Video quality:** `evaluate_visual_metrics.py`, 16 frames sampled uniformly per clip at 224×224:
  - I3D FVD, Inception FID and CLIP-FID;
  - paired SSIM, PSNR and LPIPS (AlexNet).

To rerun on the cluster:

```bash
python replication/phoenix_full642_baselines_20261103/prepare.py   # on the login node
sbatch replication/phoenix_full642_baselines_20261103/bundled.slurm  # generation + scoring, about 1 day on one A6000
bash replication/phoenix_full642_baselines_20261103/real_video/real_video_ceiling.sh  # inside a GPU allocation
```

## What is not in the repo

- Weights:
  - SignViP released checkpoints (`checkpoints/RWTH/`);
  - our FSQ and GPT checkpoints (on the cluster under `replication/final_phoenix/runs/`);
  - the CorrNet+ checkpoint;
  - I3D (`metrics/fvd/styleganv/i3d_torchscript.pt`);
  - the HF models (multilingual CLIP, OpenAI CLIP ViT-B/32).
- Data: PHOENIX-2014T processed videos, DWPose/HaMeR renders and metadata. These are under `$ROOT/signvip_dit/data/phoenix2014t/` on the cluster; their preparation is not covered here.
- Generated videos (on the cluster) and W&B logs.
- Environments: conda env `signvip` (`requirements.txt`, plus `lpips==0.1.4`) and conda env `corrnet`.

The scripts hard-code the cluster's paths (`ROOT=/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio`). Change `ROOT` at the top of each script to run them elsewhere.
