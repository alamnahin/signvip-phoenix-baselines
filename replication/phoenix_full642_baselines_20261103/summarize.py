"""Collect full-test PHOENIX-2014T results for both SignViP baselines into RESULTS.md / results.json."""
import json
from pathlib import Path

RUN = Path('/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip/replication/phoenix_full642_baselines_20261103')
SSLDIT = Path('/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio/signvip_dit/replication/phoenix_full642_20261018')

RECOG = ['bleu_1_percent', 'bleu_2_percent', 'bleu_3_percent', 'bleu_4_percent',
         'rouge_l_f1_percent', 'exact_sentence_accuracy_percent', 'wer_percent']


def flatten_quality(q):
    out = dict(q.get('metrics', {}))
    for k, v in q.get('unavailable', {}).items():
        if isinstance(v, dict) and isinstance(v.get('value'), (int, float)):
            out[k] = v['value']
    return out


def load(path, name):
    s = json.loads((path / 'corrnet/summary.json').read_text())
    q = json.loads((path / 'visual_metrics.json').read_text())
    return {'name': name, 'videos': s.get('evaluated_videos'), 'requested': s.get('requested_videos'),
            'recognition': {k: s.get(k) for k in RECOG}, 'quality': flatten_quality(q)}


# The August SSL-DiT Phoenix training (run 2026-10-18 in SSLDIT) is broken per the user, so it is left out.
rows = [load(RUN / 'released', 'SignViP released checkpoint'),
        load(RUN / 'retrained', 'SignViP corrected replication')]
for r in rows:
    assert r['videos'] == r['requested'] == 642, r

(RUN / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
qkeys = sorted({k for r in rows for k in r['quality']})
lines = ['# PHOENIX-2014T full test (642 clips), end-to-end text-to-video', '',
         'Back-translation = CorrNet+ gloss recognition (not the SignViP paper SLT model).', '',
         'Reference-frame note: in the released-checkpoint run, cv2 failed to read the first frame of 3 source '
         'videos (11January_2011_Tuesday_tagesschau-8017, 29March_2011_Tuesday_tagesschau-8108, '
         '11September_2009_Friday_tagesschau-8780), so those clips were skipped. The failure was transient: the '
         'same files decode on re-read (8/8 tries, 42.9-44.8 dB PSNR vs the original PHOENIX-2014-T '
         'images0001.png), and the retrained run read 8017 and 8108 normally. The missing clips (released: all 3; '
         'retrained: 8780 only) were generated in ref_fix/ with images0001.png as the reference frame and the same '
         'seeds, steps and CFG. Where both versions exist (retrained 8017, 8108), the PNG-reference and '
         'mp4-reference outputs agree at about 38 dB PSNR on frame 0, so the effect on the metrics is negligible.', '',
         '| Model | videos | ' + ' | '.join(RECOG + qkeys) + ' |',
         '|---|---:|' + '---:|' * (len(RECOG) + len(qkeys))]
for r in rows:
    vals = [r['recognition'].get(k) for k in RECOG] + [r['quality'].get(k) for k in qkeys]
    lines.append(f"| {r['name']} | {r['videos']} | " +
                 ' | '.join('' if v is None else f'{v:.4g}' for v in vals) + ' |')
(RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
