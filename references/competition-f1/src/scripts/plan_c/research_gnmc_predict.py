#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    models = {}
    for res in (448, 672):
        root = Path(args.model) / f'head_y_{res}'
        cfg = json.loads((root / 'config.json').read_text())
        model = SpatialHead(cfg['D']).eval()
        model.load_state_dict(torch.load(root / 'head.pt', map_location='cpu'))
        models[res] = model, cfg
    rows = list(map(json.loads, open(args.rows)))
    if args.limit:
        rows = rows[:args.limit]
    errors = []
    with open(args.out, 'w') as f:
        for i, row in enumerate(rows, 1):
            v = row['video_id']
            try:
                cs = []
                for res in (448, 672):
                    model, cfg = models[res]
                    p = Path(args.features) / f'feats_{res}' / (v + '.npz')
                    assert np.array_equal(np.load(p)['frame_indices'], row['sampling_frame_indices'])
                    cs.append(predict(model, *load_clip(p, 'y', cfg['layer_idx']), 'cpu'))
                tw, th = row['targetRatioWH']
                a = int(min(row['width'], row['height'] * tw / th)) * th / tw / row['height']
                c = np.clip(np.mean(cs, axis=0), a / 2, 1 - a / 2)
                times = np.array(row['sampling_frame_indices']) / row['fps']
                assert len(c) == len(times) and np.isfinite(c).all()
                result = {'video_id': v, 'center_y': c.tolist(), 'times': times.tolist()}
                print('GNMC PREDICT', i, len(rows), v, len(c), flush=True)
            except Exception as e:
                print('error', v, repr(e), flush=True)
                errors.append({'video_id': v, 'error': str(e)})
                result = {'video_id': v, 'predictions': []}
            f.write(json.dumps(result) + '\n')
    Path(args.out).with_suffix('.errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    print('GNMC PREDICT COMPLETE', len(rows), 'errors', len(errors), flush=True)


if __name__ == '__main__':
    main()
