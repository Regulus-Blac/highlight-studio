#!/usr/bin/env python3
"""Full16:9 learned boxes from frozen features; X1 frame selections stay fixed."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_crop_scale import (CropHead, encode, geo_features, geometry, integer_box,
                                 load_clip, load_heads, center_crop, sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--g1', type=Path, default=Path('outputs/research_44_76/gnmc_train'))
    ap.add_argument('--x1', type=Path, default=Path('outputs/research_44_76/retarget_x/train'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    assert sha(args.base) == 'fbb3fabfd438adda2788658b0f628ec555991d1da665970f423d67577da0c478'
    parent = json.loads(args.base.with_suffix('.manifest.json').read_text())
    original = parent['parameters']
    cfg = json.loads((args.model_dir / 'head.json').read_text())
    model = CropHead(cfg['dimension']).eval()
    model.load_state_dict(torch.load(args.model_dir / 'head.pt', map_location='cpu', weights_only=True))
    count = sum(p.numel() for p in model.parameters())
    assert count == cfg['parameters']
    total = original['total_parameters'] + count
    assert total <= 9000000000
    heads, inputs = load_heads(args.g1, args.x1)
    original_heads = {c['name']: c for c in original['components']}
    for res in (448, 672):
        for axis, root, prefix in [('x', args.x1, 'RetargetVid'), ('y', args.g1, 'GNMC')]:
            path = root / f'head_{axis}_{res}' / 'head.pt'
            assert sha(path) == original_heads[f'{prefix} spatial_{axis}_{res}']['files'][0]['sha256']
    rows = {r['video_id']: r for r in map(json.loads, args.rows.open())}
    assert all(r['targetRatioWH'] == [16, 9] and geometry(r['width'], r['height'], [16, 9]) is None
               for r in rows.values())
    curves, errors, features_sha = {}, [], {}
    for number, (vid, row) in enumerate(rows.items(), 1):
        try:
            expected = row['sampling_frame_indices']
            box = np.asarray(center_crop(row['width'], row['height'], 16, 9), np.int64)
            vectors = []
            for res in (448, 672):
                path = args.features / f'feats_{res}' / (vid + '.npz')
                assert np.array_equal(np.load(path)['frame_indices'], expected)
                parts = []
                for axis in ('x', 'y'):
                    frozen, conf = heads[res, axis]
                    profile, zs = load_clip(path, axis, conf['layer_idx'])
                    encoded, _ = encode(frozen, profile, zs)
                    parts.append(encoded)
                parts.append(np.tile(geo_features(row, box), (len(expected), 1)))
                vectors.append(np.concatenate(parts, axis=1))
                features_sha[str(path)] = sha(path)
            with torch.no_grad():
                residual = model(torch.from_numpy(np.stack(vectors, axis=1))).numpy()
            assert residual.shape == (len(expected), 3) and np.isfinite(residual).all()
            curves[vid] = dict(video_id=vid, times=(np.array(expected) / row['fps']).tolist(),
                               residuals=residual.tolist(), maximum_box=box.tolist())
        except Exception as e:
            print('error', vid, repr(e), flush=True)
            errors.append(dict(video_id=vid, error=str(e)))
            curves[vid] = dict(video_id=vid, predictions=[], error=str(e))
        if number % 10 == 0:
            print('FULLBOX_PREDICT', number, len(rows), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.residuals.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in curves.values()))
    changed, changed_frames, unchanged, frames = 0, 0, 0, 0
    with args.out.open('w') as stream:
        for line in args.base.open():
            row = json.loads(line)
            vid = row['video_id']
            if vid in rows:
                try:
                    curve, meta = curves[vid], rows[vid]
                    residuals = np.asarray(curve['residuals'])
                    predictions = []
                    for before in row['predictions']:
                        residual = [np.interp(before['frame'] / meta['fps'], curve['times'], residuals[:, j])
                                    for j in range(3)]
                        box = integer_box(residual, meta, curve['maximum_box']).tolist()
                        predictions.append(dict(before, bboxes=box))
                    n = sum(a['bboxes'] != b['bboxes'] for a, b in zip(row['predictions'], predictions))
                    changed += n > 0
                    changed_frames += n
                    row = dict(row, predictions=predictions)
                except Exception as e:
                    print('error', vid, repr(e), flush=True)
                    errors.append(dict(video_id=vid, error=str(e)))
                    row = dict(row, predictions=[])
            else:
                unchanged += 1
            row['model_params_b'] = total / 1e9
            frames += len(row['predictions'])
            stream.write(json.dumps(row) + '\n')
    inventory = dict(components=original['components'] + [dict(name='GNMC full16:9 center_scale',
        parameters=count, files=[dict(path=str(args.model_dir / 'head.pt'), sha256=sha(args.model_dir / 'head.pt'))])],
        total_parameters=total, model_params_b=total / 1e9, within_9b=True,
        derivation='X1 original production union + one shared full16:9 crop head; four spatial encoders reused once')
    report = dict(comparison_base='X1_retarget_x_audited', baseline_score=43.30, leaderboard_score=None,
                  parameters=inventory, source_prediction_sha256=sha(args.base), prediction_sha256=sha(args.out),
                  eligible_full_videos=len(rows), changed_full_videos=changed, changed_full_frames=changed_frames,
                  other_predictions_unchanged=unchanged, frames=frames, errors=errors,
                  algorithm='full16:9 only; equal-resolution learned dx/dy/logscale; actual-time residual interpolation -> legal integer box; X1 selected frames fixed',
                  inputs_sha256={**inputs, **{str(p): sha(p) for p in [args.base, args.rows,
                      args.model_dir / 'head.pt', args.model_dir / 'head.json', Path(__file__),
                      Path(__file__).with_name('research_crop_scale.py')]}},
                  features_sha256=features_sha, validation='NOT_RUN', submitted=False,
                  compliance='No test feedback or full92 used to fit or initialize this head. Historical submissions and inherited data/model rights still require the separate compliance audit.')
    args.out.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FULLBOX_BUILD', json.dumps({k: report[k] for k in ['eligible_full_videos', 'changed_full_videos',
          'changed_full_frames', 'other_predictions_unchanged', 'frames', 'errors', 'prediction_sha256']}), flush=True)


if __name__ == '__main__':
    main()
