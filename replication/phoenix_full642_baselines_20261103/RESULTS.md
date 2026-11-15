# PHOENIX-2014T full test (642 clips), end-to-end text-to-video

Back-translation = CorrNet+ gloss recognition (not the SignViP paper SLT model).

Reference-frame note: in the released-checkpoint run, cv2 failed to read the first frame of 3 source videos (11January_2011_Tuesday_tagesschau-8017, 29March_2011_Tuesday_tagesschau-8108, 11September_2009_Friday_tagesschau-8780), so those clips were skipped. The failure was transient: the same files decode on re-read (8/8 tries, 42.9-44.8 dB PSNR vs the original PHOENIX-2014-T images0001.png), and the retrained run read 8017 and 8108 normally. The missing clips (released: all 3; retrained: 8780 only) were generated in ref_fix/ with images0001.png as the reference frame and the same seeds, steps and CFG. Where both versions exist (retrained 8017, 8108), the PNG-reference and mp4-reference outputs agree at about 38 dB PSNR on frame 0, so the effect on the metrics is negligible.

| Model | videos | bleu_1_percent | bleu_2_percent | bleu_3_percent | bleu_4_percent | rouge_l_f1_percent | exact_sentence_accuracy_percent | wer_percent | clip_fid | fid_inception | fvd_i3d | lpips_alex | psnr_db | ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SignViP released checkpoint | 642 | 13.94 | 5.734 | 2.756 | 1.535 | 14.42 | 0.4673 | 116.2 | 3.674 | 13.91 | 87.99 | 0.3151 | 13.58 | 0.6114 |
| SignViP corrected replication | 642 | 14.32 | 6.778 | 3.809 | 2.436 | 15.45 | 0.6231 | 115.7 | 3.317 | 14.28 | 95.38 | 0.3162 | 13.61 | 0.6106 |
