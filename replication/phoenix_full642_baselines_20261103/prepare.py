"""Prepare full PHOENIX-2014T test evaluation of two SignViP baselines.

Models (no retraining, weights unchanged):
  released  - official SignViP Phoenix2014T checkpoint (checkpoints/RWTH: t2gpt.bin + FSQ inside condition_encoder.bin)
  retrained - our corrected SignViP replication (final_phoenix): FP32 FSQ fsq-job-25742 model-50000,
              GPT gpt-job-25779 model-50000 (global attention, same recipe as final Isharah baseline),
              released Stage-I generator.
Protocol matches final_phoenix job-25780 (DDIM 50, CFG 3.5) and the SSL-DiT full-642 run's metric script.
Run once on the login node before submitting bundled.slurm.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path('/SLURM/home/slurm_s202281480/karsl/DiffSynth-Studio')
SIGNVIP = ROOT / 'signvip'
FINAL = SIGNVIP / 'replication/final_phoenix'
RUN = SIGNVIP / 'replication/phoenix_full642_baselines_20261103'
DATA = ROOT / 'signvip_dit/data/phoenix2014t'
CORRNET = ROOT / 'CORRNET/CorrNet_Plus_CSLR'

RELEASED_T2S = SIGNVIP / 'configs/eval/t2s_test_phoenix100.yaml'
RELEASED_VQ = SIGNVIP / 'configs/eval/vq_test_phoenix100.yaml'
RETRAINED_T2S = FINAL / 'results/job-25780/final/t2s.yaml'
RETRAINED_VQ = FINAL / 'results/job-25780/vq.yaml'
RETRAINED_GPT = FINAL / 'runs/gpt-job-25779/workspace/model-50000/t2pgpt/model.bin'
RETRAINED_FSQ = FINAL / 'runs/fsq-job-25742/workspace/train_fsq/20260911-0014/model-50000/condition_encoder/model.bin'
RELEASED_GPT = SIGNVIP / 'checkpoints/RWTH/t2gpt.bin'
RELEASED_ENCODER = SIGNVIP / 'checkpoints/RWTH/condition_encoder.bin'
METRIC_SOURCE = FINAL / 'evaluate_visual_metrics.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    for path in [RELEASED_T2S, RELEASED_VQ, RETRAINED_T2S, RETRAINED_VQ, RETRAINED_GPT,
                 RETRAINED_FSQ, RELEASED_GPT, RELEASED_ENCODER, METRIC_SOURCE,
                 CORRNET / 'checkpoints/phoenix2014t.pt', FINAL / 'add_rouge.py']:
        if not path.is_file():
            raise FileNotFoundError(path)

    subprocess.run([
        sys.executable, str(SIGNVIP / 'prepare_phoenix2014t_eval.py'),
        '--metadata', str(DATA / 'test_meta.json'),
        '--video-root', str(DATA / 'rgb/test_processed_videos'),
        '--recognition-info', str(CORRNET / 'preprocess/phoenix2014-T/test_info.npy'),
        '--split', 'test', '--limit', '642',
        '--subset-metadata', str(RUN / 'metadata.json'),
        '--eval-spec', str(RUN / 'eval_spec.json'),
    ], check=True)
    spec = json.loads((RUN / 'eval_spec.json').read_text())
    rows = json.loads((RUN / 'metadata.json').read_text())
    assert len(rows) == len(spec['samples']) == 642, (len(rows), len(spec['samples']))
    assert len({s['id'] for s in spec['samples']}) == 642

    # Text-to-token configs: only the metadata path changes; weights and attention settings are as before.
    for name, source in [('released', RELEASED_T2S), ('retrained', RETRAINED_T2S)]:
        config = yaml.safe_load(source.read_text())
        config['dataset']['meta_paths'] = [str(RUN / 'metadata.json')]
        config['modules'].setdefault('attn_type', 'global')  # released default is global; made explicit
        (RUN / f'{name}_t2s.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for name, source in [('released', RELEASED_VQ), ('retrained', RETRAINED_VQ)]:
        (RUN / f'{name}_vq.yaml').write_text(source.read_text())

    released = yaml.safe_load((RUN / 'released_t2s.yaml').read_text())
    retrained = yaml.safe_load((RUN / 'retrained_t2s.yaml').read_text())
    assert Path(released['modules']['ckpt']) == RELEASED_GPT
    assert Path(retrained['modules']['ckpt']) == RETRAINED_GPT
    assert Path(yaml.safe_load((RUN / 'retrained_vq.yaml').read_text())['modules']['vq_model']) == RETRAINED_FSQ
    assert not yaml.safe_load((RUN / 'released_vq.yaml').read_text())['modules'].get('vq_model')

    metric_text = METRIC_SOURCE.read_text().replace(
        'SIGNVIP_ROOT = Path(__file__).resolve().parents[2]',
        f'SIGNVIP_ROOT = Path({str(SIGNVIP)!r})',
    ).replace('fixed 100-video subset', 'full 642-video test split')
    assert f'SIGNVIP_ROOT = Path({str(SIGNVIP)!r})' in metric_text
    (RUN / 'evaluate_visual_metrics.py').write_text(metric_text)

    provenance = {
        'dataset': 'PHOENIX-2014T', 'split': 'test', 'samples': 642,
        'scope': 'end-to-end text-to-token-to-video; predicted motion and duration',
        'generation': {'scheduler_steps': 50, 'guidance_scale': 3.5,
                       'seed_rule': 'per-sample generation_seed from eval_spec.json',
                       'reference_image': 'first frame of each source test video'},
        'models': {
            'released': {'gpt': str(RELEASED_GPT), 'fsq': 'inside ' + str(RELEASED_ENCODER),
                         'generator': 'released SignViP Stage I (checkpoints/RWTH)'},
            'retrained': {'gpt': str(RETRAINED_GPT), 'fsq': str(RETRAINED_FSQ),
                          'generator': 'released SignViP Stage I (checkpoints/RWTH), frozen',
                          'notes': 'FSQ 50k FP32 with fsq_fp32_no_cast.patch; GPT 50k, global attention; '
                                   'same recipe as the final Isharah SignViP baseline'}},
        'back_translation': 'CorrNet+ Phoenix2014T released checkpoint, gloss recognition '
                            '(not the SignViP paper video-to-spoken-text SLT model)',
        'video_quality': 'final_phoenix/evaluate_visual_metrics.py (I3D FVD, Inception FID, CLIP-FID, '
                         'SSIM, PSNR, LPIPS; 16 uniformly sampled frames per clip) - same script as the '
                         'SSL-DiT full-642 run of 2026-10-18',
        'sha256': {str(p): sha(p) for p in [RELEASED_GPT, RELEASED_ENCODER, RETRAINED_GPT, RETRAINED_FSQ,
                                            RUN / 'eval_spec.json', RUN / 'released_t2s.yaml',
                                            RUN / 'retrained_t2s.yaml']},
    }
    (RUN / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: v for k, v in provenance.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
