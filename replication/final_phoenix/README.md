# Final Phoenix2014T SignViP Replication

This directory records the final controlled experiment used to isolate SignViP
Stage II (FSQ) and Stage III (the multi-condition token translator). Historical
runs under the other `replication/` directories are inputs for comparison and
are not modified by this experiment.

## Fixed protocol

- Dataset: Phoenix2014T, official local train/dev/test metadata.
- Stage I: released SignViP condition encoder, appearance encoder, U-Net, motion
  module, VAE, and scheduler. Every Stage-I parameter is frozen.
- Stage II: train only a fresh FSQ autoencoder for 50,000 optimizer steps in
  FP32. The global batch is 16 across two A100s. AdamW uses `lr=5e-5`, zero
  weight decay, cosine decay, 500 warmup steps, and gradient clipping at 1.0.
- Stage II checkpoints: 10k, 20k, 30k, 40k, and 50k. Each is measured on all
  519 dev videos in condition-feature space and on a fixed first-100 dev set by
  bypassing GPT (`FSQ tokens -> FSQ decoder -> released diffusion model`).
- Stage III: extract fresh train/dev/test token files from the 50k FSQ, then
  train the released 8-block GPT architecture for 50,000 optimizer steps in
  BF16. The global batch is 64 across two A100s. AdamW8bit uses `lr=2e-4`,
  `weight_decay=1e-6`, cosine decay, 200 warmup steps, 40% input-token
  corruption, and gradient clipping at 1.0.
- Stage III validation: every 2,000 steps, record raw categorical token DTW,
  decoded-feature normalized DTW, rollout length/EOS/static-motion diagnostics,
  code usage, teacher-forced loss, and teacher-forced accuracy.
- End-to-end generation: best raw-DTW, best decoded-DTW, and final checkpoints;
  fixed first 100 test samples; fixed per-sample seeds; DDIM 50 steps; CFG 3.5.
- GPT token extraction covers train and development splits. Test FSQ tokens are
  neither available nor required: text-to-video inference predicts its tokens,
  and test labels are used only by the external evaluator.

## Result contract

The final report separates metrics by evaluator and comparability:

| Group | Metrics | Status |
|---|---|---|
| Token translator | CE, accuracy, raw token DTW, decoded-feature DTW, length/EOS/static/code statistics | Computed locally |
| Semantic proxy | CorrNet+ WER, BLEU-1/2/3/4, ROUGE-L | Computed on gloss sequences; not the paper's video-to-spoken-text model |
| Distribution quality | I3D FVD, Inception FID, CLIP-FID | Computed on the fixed 100-video subset; CLIP-FID is emitted when OpenAI CLIP ViT-B/32 is fully cached |
| Paired quality | SSIM, PSNR, LPIPS | Computed after deterministic temporal alignment; protocol is recorded |
| Paper-only | COMET, IDS, Hand SSIM | Reported as unavailable because the required translation or detector pipeline is absent; no proxy is silently substituted |

Consequently, local numbers must not be copied beside SignViP paper numbers
without the protocol column. In particular, CorrNet+ translates video to
glosses, whereas Table 1 uses a video-to-spoken-text evaluation pipeline.

## Job chain

`submit.sh` submits the following `afterok` chain:

1. `train_fsq.slurm`
2. `evaluate_fsq.slurm`
3. `train_gpt.slurm`
4. `evaluate_gpt.slurm`

Every run records its source commit, applied patches, runtime configs,
checkpoint hashes, fixed evaluation spec, and machine-readable JSON summaries.
