#!/usr/bin/env python3
"""恢复基线、动作拼接与后融合包：共享D3解码、x头、去帧及YOLO11x y框。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import torch
from safetensors import safe_open

sys.path.insert(0, 'tools')
import reframe as rf
from trim_frames import load_trim, trim_row, video_meta

T = Path('/root/sj-fs/autodl-tmp/public/aic-competition-test/2026_semifinal')
CFG = Path('tools/configs/final_AM.json')
DECODE = Path('tools/configs/d3_decode.json')
CENTERS = Path('outputs/sph_restore_avg_test_x.jsonl')
YOLOS = ['/root/sj-fs/autodl-tmp/sym/models/yolo11s.pt', '/root/sj-fs/models/yolo11x.pt']


def run(args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_count(folder):
    total = 0
    for path in Path(folder).glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as tensors:
            total += sum(math.prod(tensors.get_slice(k).get_shape()) for k in tensors.keys())
    return total


def postprocess(row, path, small, large, params, trim, stats, trim_full=None):
    """trim_full: other trim settings for videos whose source ratio already equals the target (no crop)."""
    if not row['predictions']:
        return row
    n, fps, width, height = video_meta(path)
    # the head detector only moves the final YOLO11x portrait strips, so frames and trimming never change
    pre = rf.reframe_row(row, path, dict(params, y_head_det=None), lambda: small, 'outputs/det_cache', stats)
    cache = rf.collect(path, [p['frame'] for p in pre['predictions']],
                       params['stride'], small, 'outputs/det_cache')
    geo = rf.geometry(width, height, row['targetRatioWH'])
    result, _ = trim_row(pre, cache, n, fps, trim_full if geo is None and trim_full else trim)
    if geo is not None and geo[0] == 'y':
        y = rf.reframe_row(pre, path, params, lambda: large, 'outputs/det_cache_yolo11x', stats)
        boxes = {p['frame']: p['bboxes'] for p in y['predictions']}
        result = dict(result, predictions=[dict(p, bboxes=boxes[p['frame']]) for p in result['predictions']])
    return result


def main():
    variant = sys.argv[1]
    assert variant in ['baseline', 'motion2', 'late50']
    out = Path('outputs/sub/R1T_%s_yolo11x' % variant)
    assert not out.with_suffix('.zip').exists(), '已有提交包，不覆盖'
    work = Path('outputs/motion_eval') / variant
    work.mkdir(parents=True, exist_ok=True)
    feats = 'outputs/rzf_test_timelens8b'
    if variant == 'motion2':
        feats = 'outputs/rzf_motion2_test'
        run(['tools/fuse_motion.py', 'outputs/rzf_test_timelens8b', 'outputs/motion2_test', feats])
    index = [json.loads(l) for l in open(T / 'submission_index.jsonl')]
    ids = {str(r['video_id']) for r in index}
    assert len(index) == len(ids) == 426
    inputs = [CFG, DECODE, CENTERS, T / 'submission_index.jsonl', Path(__file__),
              Path('data/train.with_video.jsonl'), Path('data/ext_mrhisum.jsonl'),
              Path('/root/sj-fs/ext/mrhisum/heldout.txt'), *map(Path, YOLOS)]
    inputs.extend(map(Path, ['tools/decode_scores.py', 'tools/reframe.py', 'tools/trim_frames.py',
                            'scripts/plan_c/ground_build.py', 'tools/train_vlm_head.py', 'tools/fuse_motion.py',
                            'tools/ensemble_probs.py']))
    parameters = {'TimeLens_8B': weight_count('/root/sj-fs/models/TimeLens-8B')}
    if variant in ['motion2', 'late50']:
        parameters['VideoMAE_K400'] = weight_count('/root/sj-fs/models/videomae-base-kinetics')
        inputs += [Path('tools/motion_feats.py'), Path('outputs/motion_preprocess_profile/motion_feats_before.py'),
                   Path('outputs/motion_preprocess_profile/cpu.json'), Path('outputs/motion_preprocess_profile/gpu.json')]
    predictions = []
    for layer in [1, 2]:
        head = Path('outputs/vh_restore_%s_L%d' % ('baseline' if variant == 'late50' else variant, layer))
        pred = work / ('test_L%d.probs.jsonl' % layer)
        run(['tools/train_vlm_head.py', '--predict', '--model-dir', head,
             '--probs', 'outputs/rz_test_timelens8b.probs.jsonl', '--feats', feats,
             '--channel', 'mix', '--device', 'cpu', '--out', pred])
        rows = [json.loads(l) for l in open(pred)]
        assert len(rows) == 426 and {r['video_id'] for r in rows} == ids
        predictions.append(pred)
        checkpoints = [head / ('fold%d.pt' % i) for i in range(5)]
        parameters['temporal_L%d_all_folds' % layer] = sum(
            t.numel() for p in checkpoints for t in torch.load(p, map_location='cpu', weights_only=True).values())
        inputs.extend([*checkpoints, head / 'config.json', pred])
    weights = [1, 1]
    if variant == 'late50':
        head = Path('outputs/vh_restore_motiononly')
        pred = work / 'test_motiononly.probs.jsonl'
        run(['tools/train_vlm_head.py', '--predict', '--model-dir', head,
             '--probs', 'outputs/motion2_test.probs.jsonl', '--feats', 'outputs/motion2_test',
             '--channel', 'mix', '--device', 'cpu', '--out', pred])
        rows = [json.loads(l) for l in open(pred)]
        assert len(rows) == 426 and {r['video_id'] for r in rows} == ids
        predictions.append(pred)
        weights.append(2)
        checkpoints = [head / ('fold%d.pt' % i) for i in range(5)]
        parameters['temporal_motiononly_all_folds'] = sum(
            t.numel() for p in checkpoints for t in torch.load(p, map_location='cpu', weights_only=True).values())
        inputs.extend([*checkpoints, head / 'config.json', pred])
    for side in [448, 672]:
        checkpoint = Path('outputs/sph_restore_%d_x/head.pt' % side)
        parameters['spatial_x_%d' % side] = sum(
            t.numel() for t in torch.load(checkpoint, map_location='cpu', weights_only=True).values())
        inputs.extend([checkpoint, checkpoint.parent / 'config.json'])
    ensemble = work / 'test.probs.jsonl'
    if variant == 'late50':
        timelens = work / 'test_timelens.probs.jsonl'
        run(['tools/ensemble_probs.py', '--probs', *predictions[:2], '--out', timelens])
        run(['tools/ensemble_probs.py', '--probs', timelens, predictions[-1], '--out', ensemble])
        inputs.append(timelens)
    else:
        run(['tools/ensemble_probs.py', '--probs', *predictions, '--weights', *weights, '--out', ensemble])
    frames = work / 'frames.jsonl'
    run(['tools/decode_scores.py', '--probs', ensemble, '--params', DECODE,
         '--index', T / 'submission_index.jsonl', '--video-dir', T / 'video',
         '--safe-frames', '--write-submission', frames])

    cfg = json.load(open(CFG))
    params, trim = rf.load_params(**cfg['reframe']), load_trim(cfg['trim'])
    small, large = [rf.Detector(path, None, cfg['imgsz']) for path in YOLOS]
    for name, det in [('yolo11s', small), ('yolo11x', large)]:
        parameters[name] = sum(p.numel() for p in det.model.model.parameters())
    assert sum(parameters.values()) <= 9_000_000_000, parameters
    stats = Counter()
    stats['_offsets'] = []
    errors = 0
    before_x = work / 'trimmed_yolo11x.jsonl'
    with open(before_x, 'w') as f:
        for i, row in enumerate(map(json.loads, open(frames)), 1):
            result = row
            try:
                result = postprocess(row, str(T / 'video' / (row['video_id'] + '.mp4')),
                                     small, large, params, trim, stats)
            except Exception as error:
                print('error video=%s: %s' % (row['video_id'], error), flush=True)
                result, errors = dict(row, predictions=[]), errors + 1
            f.write(json.dumps(result) + '\n')
            f.flush()
            if i % 25 == 0:
                print('postprocess %d/426' % i, flush=True)
    assert errors == 0, '%d videos failed; no package' % errors
    run(['scripts/plan_c/ground_build.py', before_x, CENTERS, out.with_suffix('.jsonl')])
    command = [sys.executable, 'tools/validate_submission.py', '--pred', str(out.with_suffix('.jsonl')),
               '--index', str(T / 'submission_index.jsonl'), '--video-dir', str(T / 'video')]
    checked = subprocess.run(command, text=True, capture_output=True)
    out.with_suffix('.validate.log').write_text(checked.stdout + checked.stderr)
    print(checked.stdout + checked.stderr, flush=True)
    checked.check_returncode()
    with zipfile.ZipFile(out.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out.with_suffix('.jsonl'), arcname='predictions.jsonl')
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:
        assert z.namelist() == ['predictions.jsonl']
        assert z.read('predictions.jsonl') == out.with_suffix('.jsonl').read_bytes()
    rows = [json.loads(l) for l in open(out.with_suffix('.jsonl'))]
    manifest = {'variant': variant, 'sha256': digest(out.with_suffix('.zip')),
                'jsonl_sha256': digest(out.with_suffix('.jsonl')), 'validation': 'PASS',
                'videos': len(rows), 'frames': sum(len(r['predictions']) for r in rows),
                'parameters': parameters, 'total_parameters': sum(parameters.values()),
                'probability_weights': [w / sum(weights) for w in weights],
                'comparison': '恢复版基线对动作融合版；不是39.44包的逐字节复现。',
                'fixed': '原800 MRH及100留出、D3解码、448/672 x头、final_AM去帧、YOLO11x y框。',
                'note': '选帧变化会影响y跟踪输入；规则固定，不声称所有共有帧的y框相同。',
                'validation_command': command, 'input_sha256': {str(p): digest(p) for p in inputs},
                'expected_score': None, 'submitted': False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('MOTION_PACKAGE_DONE', out, manifest['total_parameters'], flush=True)


if __name__ == '__main__':
    main()
