#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip, medfilt, to_t


@torch.no_grad()
def readout(model, prof, zs):
    latent, stats = [], []
    for start in range(0, len(prof), 128):
        batch = [(prof[i], zs[i], 0., .5) for i in range(start, min(start + 128, len(prof)))]
        x, pos, box, _, _ = to_t(batch, 'cpu')
        h = model.enc(model.proj(x) + model.pos(pos[..., None]) + model.box(box)[:, None])
        p = model.out(h).squeeze(-1).softmax(-1)
        c = (p * pos).sum(-1)
        sd = (p * (pos - c[:, None]).square()).sum(-1).sqrt()
        entropy = -(p * p.clamp(min=1e-10).log()).sum(-1) / np.log(p.shape[1])
        latent.append((p[..., None] * h).sum(1).numpy())
        stats.append(torch.stack([c, sd, entropy, p.max(-1).values], -1).numpy())
    return np.concatenate(latent), np.concatenate(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dense', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(map(json.loads, open(args.rows)))
    if args.limit:
        rows = rows[:args.limit]
    models = {}
    for axis in ('x', 'y'):
        for res in (448, 672):
            root = base / f'head_{axis}_{res}'
            cfg = json.loads((root / 'config.json').read_text())
            model = SpatialHead(cfg['D']).eval()
            model.load_state_dict(torch.load(root / 'head.pt', map_location='cpu'))
            models[axis, res] = (model, cfg['layer_idx'])
    reference = {} if args.dense else {r['video_id']: r for r in map(json.loads, (base / 'predictions.jsonl').open())}
    for n, row in enumerate(rows, 1):
        v = row['video_id']
        dest = out / (v + '.npz')
        if dest.exists():
            continue
        try:
            axis = 'x' if row['width'] / row['height'] > row['targetRatioWH'][0] / row['targetRatioWH'][1] else 'y'
            vectors, stats, centers = [], [], []
            for res in (448, 672):
                model, li = models[axis, res]
                latent, stat = readout(model, *load_clip(base / f'feats_{res}' / (v + '.npz'), axis, li))
                vectors.append(latent)
                stats.append(stat)
                centers.append(stat[:, 0] if args.dense else np.array([round(float(c), 4) for c in medfilt(stat[:, 0])]))
            c = np.mean(centers, axis=0)
            if not args.dense:
                assert np.max(np.abs(c - reference[v]['center_' + axis])) < 1e-7, v
            motion = np.gradient(c) if len(c) > 1 else np.zeros(1)
            geo = np.column_stack([centers[0], centers[1], *[s[:, i] for s in stats for i in range(4)], motion])
            times = np.array(row['sampling_frame_indices']) / row['fps'] if args.dense else np.arange(len(c))
            np.savez(dest, center=c, geo=geo, latent=np.concatenate(vectors, axis=1), axis=axis, times=times)
            print(n, len(rows), v, axis, len(c), flush=True)
        except Exception as e:
            print('error', v, repr(e), flush=True)
            with (out / 'errors.jsonl').open('a') as f:
                f.write(json.dumps({'video_id': v, 'predictions': [], 'error': str(e)}) + '\n')
    print('RESIDUAL FEATURES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
