#!/usr/bin/env python3
"""Recompute the learned full boxes and verify every X1 frame and untouched row."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from research_crop_scale import (CropHead, center_crop, encode, geo_features, geometry,
                                 integer_box, load_clip, load_heads, sha)


ap = argparse.ArgumentParser()
ap.add_argument('--base', type=Path, required=True)
ap.add_argument('--rows', type=Path, required=True)
ap.add_argument('--features', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--package', type=Path, required=True)
ap.add_argument('--g1', type=Path, default=Path('outputs/research_44_76/gnmc_train'))
ap.add_argument('--x1', type=Path, default=Path('outputs/research_44_76/retarget_x/train'))
args = ap.parse_args()
torch.set_num_threads(4)
assert sha(args.base) == 'fbb3fabfd438adda2788658b0f628ec555991d1da665970f423d67577da0c478'
old_manifest = json.loads(args.base.with_suffix('.manifest.json').read_text())
manifest = json.loads(args.package.with_suffix('.manifest.json').read_text())
inventory = manifest['parameters']
cfg = json.loads((args.model_dir / 'head.json').read_text())
model = CropHead(cfg['dimension']).eval()
model.load_state_dict(torch.load(args.model_dir / 'head.pt', map_location='cpu', weights_only=True))
count = sum(p.numel() for p in model.parameters())
assert count == cfg['parameters'] == 102559
assert inventory['total_parameters'] == old_manifest['parameters']['total_parameters'] + count == 8926602839
assert inventory['total_parameters'] <= 9000000000
assert sum(c['parameters'] for c in inventory['components']) == inventory['total_parameters']
assert inventory['components'][:-1] == old_manifest['parameters']['components']
assert inventory['components'][-1]['files'][0]['sha256'] == sha(args.model_dir / 'head.pt')
assert inventory['components'][-1]['parameters'] == count
assert not manifest['errors'] and manifest['validation'] == 'PASS'
pred_path = args.package.with_suffix('.jsonl')
assert manifest['prediction_sha256'] == sha(pred_path)
with zipfile.ZipFile(args.package.with_suffix('.zip')) as z:
    assert z.namelist() == ['predictions.jsonl'] and z.read('predictions.jsonl') == pred_path.read_bytes()
read = lambda p: {r['video_id']: r for r in map(json.loads, p.open())}
before, after = read(args.base), read(pred_path)
rows = read(args.rows)
curves = read(args.package.with_suffix('.residuals.jsonl'))
all_rows_path = args.rows.with_suffix('.all_metadata.jsonl')
metadata = read(all_rows_path)
eligible = {v for v, r in metadata.items() if r['targetRatioWH'] == [16, 9]
            and geometry(r['width'], r['height'], [16, 9]) is None}
assert set(before) == set(after) == set(metadata) and len(before) == 426
assert set(rows) == set(curves) == eligible and len(rows) == 167
heads, head_sha = load_heads(args.g1, args.x1)
feature_sha, changed, changed_frames, total_frames = {}, 0, 0, 0
for vid, old in before.items():
    new = after[vid]
    assert new['model_params_b'] == inventory['model_params_b']
    assert {k: v for k, v in old.items() if k not in ('model_params_b', 'predictions')} == {
        k: v for k, v in new.items() if k not in ('model_params_b', 'predictions')}
    assert [p['frame'] for p in old['predictions']] == [p['frame'] for p in new['predictions']]
    total_frames += len(new['predictions'])
    if vid not in rows:
        assert old['predictions'] == new['predictions']
        continue
    row, curve = rows[vid], curves[vid]
    box = np.asarray(center_crop(row['width'], row['height'], 16, 9), np.int64)
    assert curve['maximum_box'] == box.tolist()
    assert np.array_equal(curve['times'], np.asarray(row['sampling_frame_indices']) / row['fps'])
    vectors = []
    for res in (448, 672):
        path = args.features / f'feats_{res}' / (vid + '.npz')
        assert np.array_equal(np.load(path)['frame_indices'], row['sampling_frame_indices'])
        parts = []
        for axis in ('x', 'y'):
            frozen, conf = heads[res, axis]
            profile, zs = load_clip(path, axis, conf['layer_idx'])
            encoded, _ = encode(frozen, profile, zs)
            parts.append(encoded)
        parts.append(np.tile(geo_features(row, box), (len(curve['times']), 1)))
        vectors.append(np.concatenate(parts, axis=1))
        feature_sha[str(path)] = sha(path)
    with torch.no_grad():
        expected = model(torch.from_numpy(np.stack(vectors, axis=1))).numpy()
    assert np.array_equal(curve['residuals'], expected), vid
    n = 0
    for a, b in zip(old['predictions'], new['predictions']):
        assert {k: v for k, v in a.items() if k != 'bboxes'} == {k: v for k, v in b.items() if k != 'bboxes'}
        residual = [np.interp(a['frame'] / row['fps'], curve['times'], expected[:, j]) for j in range(3)]
        assert b['bboxes'] == integer_box(residual, row, box).tolist()
        n += a['bboxes'] != b['bboxes']
    changed += n > 0
    changed_frames += n
assert total_frames == 148810 and changed > 0
for old_line, new_line in zip(args.base.read_text().splitlines(), pred_path.read_text().splitlines()):
    old = json.loads(old_line)
    if old['video_id'] not in rows:
        assert old_line.replace('"model_params_b": ' + str(old['model_params_b']),
                                '"model_params_b": ' + str(inventory['model_params_b'])) == new_line
assert feature_sha == manifest['features_sha256']
audit = dict(status='PASS', videos=426, selected_full16_9_videos=167,
             other259_predictions_identical=True, other259_lines_only_parameter_metadata_changed=True,
             unsupported_full9_16_videos_preserved=5, all_frames_unchanged=True, frames=total_frames,
             changed_videos=changed, changed_frames=changed_frames, full_residuals_recomputed_exact=True,
             all_integer_boxes_recomputed_exact=True, all_original_components_preserved=True,
             new_head_parameters=count, total_parameters=inventory['total_parameters'],
             head_sha256=head_sha, feature_sha256=feature_sha, prediction_sha256=sha(pred_path),
             zip_sha256=sha(args.package.with_suffix('.zip')), parent_prediction_sha256=sha(args.base),
             audit_code_sha256=sha(__file__), submitted=False,
             compliance='Technical checks only; inherit the separately recorded rule/license issues.')
args.package.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print('FULLBOX_PACKAGE_AUDIT_PASS', json.dumps({k: v for k, v in audit.items()
      if k not in ('head_sha256', 'feature_sha256')}), flush=True)
