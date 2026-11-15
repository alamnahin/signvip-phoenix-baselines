# Final Phoenix2014T Results

These fixed-subset local scores are not directly comparable to the full-test SignViP paper tables. CorrNet+ evaluates gloss recognition, not video-to-spoken-text translation.

| Selection | Step | Coverage | WER | B1 | B2 | B3 | B4 | ROUGE-L | COMET | FID | CLIP-FID | FVD | IDS | SSIM | PSNR | LPIPS | Hand SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best_raw_dtw | 2000 | 100.000 | 100.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N/A | 79.711 | 10.485 | 939.440 | N/A | 0.642 | 13.795 | 0.298 | N/A |
| best_decoded_dtw | 2000 | 100.000 | 100.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N/A | 79.973 | 10.547 | 942.116 | N/A | 0.642 | 13.795 | 0.298 | N/A |
| final | 50000 | 100.000 | 122.989 | 12.940 | 5.899 | 3.295 | 2.069 | 14.982 | N/A | 23.612 | 4.348 | 246.673 | N/A | 0.616 | 13.790 | 0.311 | N/A |

All semantic percentage columns use CorrNet+ gloss hypotheses. FID, FVD,
SSIM, PSNR, and LPIPS use the explicit protocol in each `visual_metrics.json`.
Unavailable metrics are deliberately not replaced with incompatible proxies.
