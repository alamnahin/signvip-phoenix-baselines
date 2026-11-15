# PHOENIX-2014T baselines: scores

*All 642 test clips · one run per model · 8 Nov 2026*

| | WER ↓ | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | exact % | FVD ↓ | FID ↓ | CLIP-FID ↓ | SSIM | PSNR | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real test video | 21.90 | 80.85 | 72.78 | 65.64 | 59.29 | 81.55 | 26.64 | – | – | – | – | – | – |
| SignViP released | 116.16 | 13.94 | 5.73 | 2.76 | 1.53 | 14.42 | 0.47 | 87.99 | 13.91 | 3.67 | 0.611 | 13.58 | 0.315 |
| SignViP retrained | 115.71 | 14.32 | 6.78 | 3.81 | 2.44 | 15.45 | 0.62 | 95.38 | 14.28 | 3.32 | 0.611 | 13.61 | 0.316 |

| Glosses (4,264 in the references) | correct | substituted | deleted | inserted |
|---|---:|---:|---:|---:|
| Real test video | 3,433 | 569 | 262 | 103 |
| SignViP released | 474 | 3,242 | 548 | 1,163 |
| SignViP retrained | 494 | 3,189 | 581 | 1,164 |

- **WER, BLEU, ROUGE-L, exact %:** gloss recognition by CorrNet+, using its released PHOENIX-2014T checkpoint with greedy decoding. This is not the SignViP paper's video-to-text model, so these numbers are not comparable to its Table 1.
- **FVD, FID, CLIP-FID, SSIM, PSNR, LPIPS:** computed on 16 frames per clip, 224×224.
- **Code:** https://github.com/alamnahin/signvip-phoenix-baselines (private).
