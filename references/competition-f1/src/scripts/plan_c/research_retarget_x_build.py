#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from highlight.geometry import safe_crop_size, dof_range, place_1d


ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--rows', required=True)
ap.add_argument('--centers', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
assert hashlib.sha256(Path(args.base).read_bytes()).hexdigest() == '8acbb67bc03bc351b0591c8282823c757b218f9bbe0c1fb9e57318bee1188dd7'
rows = {str(r['video_id']): r for r in map(json.loads, open(args.rows))}
centers = {str(r['video_id']): r for r in map(json.loads, open(args.centers))}
assert set(rows) == set(centers)
assert all(r['width'] / r['height'] > r['targetRatioWH'][0] / r['targetRatioWH'][1] for r in rows.values())
changed, changed_frames, unchanged, errors = 0, 0, 0, []
with open(args.out, 'w') as f:
    for line in open(args.base):
        row = json.loads(line)
        vid = str(row['video_id'])
        if vid not in rows:
            f.write(line)
            unchanged += 1
            continue
        try:
            m, c = rows[vid], centers[vid]
            assert len(c['times']) == len(c['center_x']) and np.isfinite(c['center_x']).all()
            tw, th = row['targetRatioWH']
            crop_w, _ = safe_crop_size(m['width'], m['height'], tw, th)
            travel = dof_range(m['width'], m['height'], tw, th)
            preds = []
            for p in row['predictions']:
                cx = np.interp(p['frame'] / m['fps'], c['times'], c['center_x'])
                _, y, w = p['bboxes']
                x, y0, w0 = place_1d((cx * m['width'] - crop_w / 2) / travel,
                                     m['width'], m['height'], tw, th)
                assert w == w0 and y == y0
                preds.append(dict(p, bboxes=[x, y, w]))
            changes = sum(p['bboxes'] != q['bboxes'] for p, q in zip(row['predictions'], preds))
            changed += changes > 0
            changed_frames += changes
            row = dict(row, predictions=preds)
        except Exception as e:
            print('error', vid, repr(e), flush=True)
            errors.append(dict(video_id=vid, error=str(e)))
            row = dict(row, predictions=[])
        f.write(json.dumps(row) + '\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = dict(changed_x_videos=changed, changed_x_frames=changed_frames,
              other_rows_byte_identical_before_parameter_metadata=unchanged, errors=errors,
              source_sha256=sha(args.base), prediction_sha256=sha(args.out))
Path(args.out).with_suffix('.change.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
