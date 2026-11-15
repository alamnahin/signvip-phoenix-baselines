# CorrNet+ (the back-translation evaluator)

The scores use [CorrNet+](https://github.com/hulianyuyy/CorrNet_Plus) at commit
`f58b4ede6aa463ce95e0c4e8f34629873d5ecef3` (see `base_commit.txt`) with its released PHOENIX-2014T
checkpoint (`CorrNet_Plus_CSLR/checkpoints/phoenix2014t.pt`, sha256 `10b71dc4…0d0`) and gloss dictionary
(`CorrNet_Plus_CSLR/preprocess/phoenix2014-T/gloss_dict.npy`). CorrNet+ is not vendored here.

`local_changes.patch` is the cluster checkout's `git diff` against that commit. Apply it with
`git -C <CorrNet_Plus checkout> apply local_changes.patch`.

What matters for the scores:

- **Greedy decoding.** `slr_network.py` builds its decoder with `'max'` (best path) instead of `'beam'`
  (beam width 10), and `utils/decode.py` imports `ctcdecode` only for beam search. `evaluate_corrnet.py`
  reads `recognized_sents` from this decoder, so every WER and BLEU number here uses greedy CTC decoding.
- `utils/video_augmentation.py`: `TemporalRescale` takes an optional `max_len`. This is training-time
  augmentation and is not used at evaluation.

The other changes (dataset root, `dataloader_video.py`, `main.py`, `seq_scripts.py`) are for training
CorrNet+ on Isharah-1000. The PHOENIX evaluator does not use them: `evaluate_corrnet.py` loads
the videos and the model itself.
