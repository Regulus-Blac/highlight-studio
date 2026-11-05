#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--centers', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = {r['video_id']: r for r in map(json.loads, open(args.rows))}
    centers = {r['video_id']: r for r in map(json.loads, open(args.centers))}
    changed, unchanged, errors = 0, 0, []
    with open(args.out, 'w') as f:
        for line in open(args.base):
            row = json.loads(line)
            v = str(row['video_id'])
            if v not in rows:
                f.write(line)
                unchanged += 1
                continue
            try:
                m, c = rows[v], centers[v]
                old = row['predictions']
                preds = []
                for p in old:
                    cx = float(np.interp(p['frame'] / m['fps'], c['times'], c['center_y']))
                    x, _, w = p['bboxes']
                    h = w * row['targetRatioWH'][1] / row['targetRatioWH'][0]
                    y = min(max(0, int(round(cx * m['height'] - h / 2))), int(m['height'] - h))
                    preds.append(dict(p, bboxes=[x, y, w]))
                assert [p['frame'] for p in old] == [p['frame'] for p in preds]
                changed += preds != old
                row = dict(row, predictions=preds)
            except Exception as e:
                print('error', v, repr(e), flush=True)
                errors.append({'video_id': v, 'error': str(e)})
                row = dict(row, predictions=[])
            f.write(json.dumps(row) + '\n')
    report = {'changed_y_videos': changed, 'other_rows_byte_identical_before_global_parameter_metadata': unchanged,
              'errors': errors, 'source_sha256': hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}
    Path(args.out).with_suffix('.change.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
