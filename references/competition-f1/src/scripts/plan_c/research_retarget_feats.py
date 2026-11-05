#!/usr/bin/env python3
"""Exact canonical 1FPS, decoder18 profiles for qualified RetargetVid200."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import spatial_feats

DATA = Path('outputs/research_44_76/retargetvid_audit/full')
OUT = Path('outputs/research_44_76/retarget_x')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-side', type=int, choices=[448, 672], required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    ready = json.load((DATA / 'READY.json').open())
    assert ready['status'] == 'READY' and ready['split_counts'] == dict(train=160, dev=20, heldout=20)
    rows_path = DATA / 'all_clean.jsonl'
    assert sha(rows_path) == ready['all_clean_sha256']
    rows = [json.loads(line) for line in rows_path.open()]
    if args.smoke:
        rows = [next(r for r in rows if r['dataset_split'] == s) for s in ['train', 'dev']]
    for row in rows:
        expected = [round(k * row['fps']) for k in range(int(np.floor((row['n_frames'] - 1) / row['fps'])) + 1)]
        assert expected == row['sampling_frame_indices']
    # The shared 4090D must be free immediately before loading the only 8B model.
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not active, active
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    selected_path = args.out / ('feature_rows' + suffix + '.jsonl')
    selected_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    folder = args.out / f'feats_{args.max_side}'
    started = time.time()
    sys.argv = ['spatial_feats.py', '--model', args.model, '--gt', str(selected_path),
                '--out-dir', str(folder), '--max-side', str(args.max_side), '--batch', '4',
                '--layers', '18', '--annotated-frames', '--no-generate', '--device', 'cuda:0']
    spatial_feats.main()
    files = {}
    for row in rows:
        path = folder / (row['video_id'] + '.npz')
        z = np.load(path)
        assert np.array_equal(z['frame_indices'], row['sampling_frame_indices'])
        assert z['col'].shape[0] == len(row['sampling_frame_indices'])
        assert z['col'].shape[1] == 1 and z['col'].shape[-1] == 4096
        assert np.isfinite(z['col']).all() and np.isfinite(z['row']).all()
        assert np.isnan(z['box']).all()
        files[str(path)] = sha(path)
    report = dict(sources=len(rows), sampled_frames=sum(len(r['sampling_frame_indices']) for r in rows),
                  max_side=args.max_side, decoder_layer=18, layer_idx=0, precision='frozen base BF16 forward / FP16 cache',
                  adapter_used=False, no_generate=True, batch=4, seconds=time.time() - started,
                  original_frame_indices_exact=True, no_test_or_blind=True,
                  inputs={str(p): sha(p) for p in [Path(__file__), Path(spatial_feats.__file__),
                                                 DATA / 'READY.json', selected_path]}, output_sha256=files)
    (args.out / ('extract_%d%s.json' % (args.max_side, suffix))).write_text(json.dumps(report, indent=2) + '\n')
    print('RETARGET_FEATURES_COMPLETE', json.dumps({k: v for k, v in report.items() if k not in ['inputs', 'output_sha256']}), flush=True)


if __name__ == '__main__':
    main()
