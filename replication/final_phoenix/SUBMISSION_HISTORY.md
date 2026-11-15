# Submission History

## 2026-09-11

- Job `25737` requested four A100s and failed during the initial diffusion
  preview because physical GPU 3 already contained an untracked process using
  about 67 GiB. The SignViP rank itself used about 11.4 GiB. No optimizer step
  or checkpoint from this attempt is used.
- Jobs `25738`-`25740` were cancelled after their dependency became impossible.
- The Stage-II layout was returned to the previously validated two-A100 setup:
  batch 8 per GPU, global batch 16. This changes no optimization hyperparameter.
- Active replacement chain: FSQ train `25742`, FSQ oracle evaluation `25743`,
  GPT train `25744`, and final evaluation `25745`.
- FSQ train `25742` and oracle evaluation `25743` completed successfully.
  GPT job `25744` stopped before optimization because its orchestration tried to
  extract unused test FSQ tokens, while Phoenix test DWPose/HAMER inputs are not
  present. Final evaluation `25745` therefore had an unsatisfied dependency.
  Test-token extraction and verification were removed; GPT training and final
  inference require only train/dev FSQ tokens and test text metadata.
- Replacement GPT job `25779` uses FSQ train `25742` and oracle evaluation
  `25743`. Final evaluation job `25780` depends on successful GPT completion.
