"""Original y-head versus video balancing / frozen layer fusion, with a sealed source holdout.

只在来源隔离的训练标签上选择；fold4 不进入开发阶段训练。测试特征只用于最终自动推理。
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import train_spatial_head as sh

OUT = ROOT / 'outputs/experiments/sym42_yopt'
ARMS = ['baseline', 'balanced', 'fused_balanced']
PARAMS = {'epochs': 15, 'batch': 128, 'lr': 5e-4, 'weight_decay': 0.05,
          'hidden': 256, 'encoder_layers': 2, 'heads': 4, 'final_seed': 99}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class MaskedHead(sh.SpatialHead):
    # Same parameters as the source head; mask only duplicated padding cells.
    def forward(self, x, pos, box, mask=None):
        h = self.proj(x) + self.pos(pos[..., None]) + self.box(box)[:, None, :]
        logits = self.out(self.enc(h, src_key_padding_mask=mask)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)
        return (logits.softmax(-1) * pos).sum(-1)


def fused_profile(row):
    # Normalize each frozen layer per cell before a fixed equal-weight fusion.
    a = row.astype(np.float32)
    a -= a.mean(axis=-1, keepdims=True)
    a /= np.sqrt((a * a).mean(axis=-1, keepdims=True) + 1e-5)
    return a.mean(axis=1)


def load(rows, side, smoke=False):
    data = {}
    cells = {}
    for i, r in enumerate(rows):
        v = r['video_id']
        with np.load(ROOT / ('outputs/spf_yhead_%s' % side) / (v + '.npz')) as z:
            row = z['row']
            base = row[:, 0].astype(np.float32)
            fused = fused_profile(row)
            assert np.isnan(z['box']).all()
        trajectory = r['teacher_signals']['trajectory']
        assert len(base) == len(trajectory), v
        y = np.array([p['center'][1] for p in trajectory], dtype=np.float32)
        cells[str(base.shape[1])] = cells.get(str(base.shape[1]), 0) + 1
        data[v] = {'base': base, 'fused': fused, 'zs': np.tile(np.array([.5, 0], np.float32), (len(y), 1)),
                   'y': y, 'group': r['provenance']['source_group'], 'source': r['provenance']['source']}
        if (i + 1) % 400 == 0:
            print('LOAD', side, i + 1, flush=True)
    print('LOADED', side, len(data), 'row_cells', cells, flush=True)
    if not smoke:
        write(OUT / ('cache_shapes_%d.json' % side), {'videos': len(data), 'row_cells': cells, 'seconds': sum(len(d['y']) for d in data.values())})
    return data


def profile(d, arm):
    return d['fused' if arm == 'fused_balanced' else 'base']


def fit(data, ids, arm, seed, target, epochs=15):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = (sh.SpatialHead if arm == 'baseline' else MaskedHead)(4096).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'], weight_decay=PARAMS['weight_decay'])
    seconds = sum(len(data[v]['y']) for v in ids)
    items = [(profile(data[v], arm)[s], data[v]['zs'][s], data[v]['y'][s], seconds / (len(ids) * len(data[v]['y'])))
             for v in ids for s in range(len(data[v]['y']))]
    started = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.
        for batch in sh.batches(items, PARAMS['batch'], rng):
            X, P, B, y = sh.to_t([b[:3] for b in batch], 'cuda')
            if arm == 'baseline':
                pred = model(X, P, B)
                loss = (pred - y).abs().mean()
            else:
                sizes = torch.tensor([b[0].shape[0] for b in batch], device='cuda')
                mask = torch.arange(X.shape[1], device='cuda')[None] >= sizes[:, None]
                pred = model(X, P, B, mask=mask)
                weights = torch.tensor([b[3] for b in batch], dtype=torch.float32, device='cuda')
                loss = ((pred - y).abs() * weights).mean()
            assert torch.isfinite(loss), 'non-finite training loss'
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('TRAIN', target.name, arm, 'epoch', epoch + 1, 'loss', round(total / seconds, 6), 'elapsed', round(time.time() - started, 1), flush=True)
    target.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), target / 'head.pt')
    write(target / 'config.json', dict(PARAMS, D=4096, axis='y', layer_idx=0, arm=arm,
          feature_mode='mean_ln_18_27' if arm == 'fused_balanced' else 'layer18',
          mask_padding=arm != 'baseline', seed=seed, epochs=epochs,
          training_video_ids=ids, training_seconds=seconds, train_sec=round(time.time() - started, 1)))
    return model


def predict(model, data, ids, arm):
    return {v: sh.predict(model, profile(data[v], arm), data[v]['zs'], 'cuda').astype(float).tolist() for v in ids}


def ensemble(a, b):
    # Source inference: each side median -> side mean -> ground_build median.
    return sh.medfilt((sh.medfilt(np.array(a)) + sh.medfilt(np.array(b))) / 2)


def video_errors(rows, pa, pb):
    errors = {}
    for r in rows:
        v = r['video_id']
        target = np.array([p['center'][1] for p in r['teacher_signals']['trajectory']])
        pred = ensemble(pa[v], pb[v])
        assert len(target) == len(pred) and np.isfinite(pred).all()
        errors[v] = float(np.abs(target - pred).mean())
    return errors


def comparison(rows, base, candidate):
    vids = sorted(base)
    groups = {}
    for r in rows:
        v = r['video_id']
        if v in base:
            groups.setdefault(r['provenance']['source_group'], []).append(candidate[v] - base[v])
    # Resample whole source groups, preserving the per-video metric.
    sums = np.array([sum(x) for x in groups.values()])
    counts = np.array([len(x) for x in groups.values()])
    indices = np.random.default_rng(42).integers(0, len(groups), size=(10000, len(groups)))
    delta = sums[indices].sum(1) / counts[indices].sum(1)
    return {'baseline_mae': float(np.mean([base[v] for v in vids])),
            'candidate_mae': float(np.mean([candidate[v] for v in vids])),
            'delta_candidate_minus_baseline': float(np.mean([candidate[v] - base[v] for v in vids])),
            'source_bootstrap_delta_95ci': np.quantile(delta, [.025, .975]).tolist(),
            'videos': len(vids), 'groups': len(groups),
            'by_source': {source: {'videos': len(values), 'delta': float(np.mean(values))}
                          for source in ['tiktok', 'pexels']
                          if (values := [candidate[r['video_id']] - base[r['video_id']] for r in rows
                                         if r['video_id'] in base and r['provenance']['source'] == source])}}


def smoke(rows):
    available = [r for r in rows if (ROOT / 'outputs/spf_yhead_448' / (r['video_id'] + '.npz')).exists()]
    assert len(available) >= 2
    data = load(available[:2], 448, smoke=True)
    ids = sorted(data)
    for arm in ARMS:
        m = fit(data, ids, arm, 0, OUT / 'smoke' / arm, epochs=1)
        p = predict(m, data, ids, arm)
        assert all(np.isfinite(x).all() for x in p.values())
        del m
    print('SMOKE_PASS', flush=True)
    write(OUT / 'smoke_complete.json', {'passed': True, 'arms': ARMS, 'video_ids': ids})


def run(rows):
    groups = sorted({r['provenance']['source_group'] for r in rows})
    random.Random(0).shuffle(groups)
    folds = {g: i % 5 for i, g in enumerate(groups)}
    dev = [r for r in rows if folds[r['provenance']['source_group']] != 4]
    holdout = [r for r in rows if folds[r['provenance']['source_group']] == 4]
    write(OUT / 'protocol.json', {'params': PARAMS, 'arms': ARMS, 'development_rows': len(dev), 'holdout_rows': len(holdout),
          'development_training_excludes_holdout': True, 'fold_by_group': folds,
          'selection_metric': 'video-macro center MAE to supplied teacher, median(side median mean)',
          'promotion': 'choose lowest development ensemble MAE; require holdout paired source-bootstrap 95CI entirely below zero; otherwise baseline',
          'teacher_metric_is_not_competition_score': True, 'test_used_for_selection': False,
          'label_sha256': hashlib.sha256((ROOT / 'data/ybox_train_v3.jsonl').read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch_version': torch.__version__})
    all_oof = {arm: {} for arm in ARMS}
    for side in [448, 672]:
        data = load(dev, side)
        ids = sorted(data)
        for arm in ARMS:
            preds = {}
            for k in range(4):
                train = [v for v in ids if folds[data[v]['group']] != k]
                val = [v for v in ids if folds[data[v]['group']] == k]
                m = fit(data, train, arm, k, OUT / ('dev_%s_%s_fold%d' % (arm, side, k)))
                preds.update(predict(m, data, val, arm))
                del m
            all_oof[arm][side] = preds
            write(OUT / ('dev_%s_%s_predictions.json' % (arm, side)), preds)
        del data
        gc.collect()
    dev_errors = {arm: video_errors(dev, all_oof[arm][448], all_oof[arm][672]) for arm in ARMS}
    chosen = min(ARMS, key=lambda arm: np.mean(list(dev_errors[arm].values())))
    report = {'chosen_on_development': chosen,
              'development': {arm: comparison(dev, dev_errors['baseline'], dev_errors[arm]) for arm in ARMS}}
    write(OUT / 'development_selection.json', report)
    print('DEVELOPMENT_SELECTION', json.dumps(report), flush=True)
    # The holdout has never been loaded during development training or selection.
    held = {arm: {} for arm in dict.fromkeys(['baseline', chosen])}
    dev_ids = sorted(r['video_id'] for r in dev)
    held_ids = sorted(r['video_id'] for r in holdout)
    for side in [448, 672]:
        data = load(rows, side)
        for arm in held:
            m = fit(data, dev_ids, arm, 99, OUT / ('holdout_%s_%s' % (arm, side)))
            held[arm][side] = predict(m, data, held_ids, arm)
            write(OUT / ('holdout_%s_%s_predictions.json' % (arm, side)), held[arm][side])
            del m
        del data
        gc.collect()
    base = video_errors(holdout, held['baseline'][448], held['baseline'][672])
    cand = video_errors(holdout, held[chosen][448], held[chosen][672])
    confirmation = comparison(holdout, base, cand)
    promoted = chosen != 'baseline' and confirmation['source_bootstrap_delta_95ci'][1] < 0
    winner = chosen if promoted else 'baseline'
    report.update(holdout=confirmation, promoted=bool(promoted), final_arm=winner,
                  limitation='Teacher agreement only; no hidden-test/leaderboard score measured.')
    write(OUT / 'selection.json', report)
    print('HOLDOUT_CONFIRMATION', json.dumps(report), flush=True)
    # A single full-data seed99 fit per resolution, as in sym; only the validated arm is exported.
    files = []
    test_ids = [json.loads(line)['video_id'] for line in (ROOT / 'data/ytest_index.jsonl').read_text().splitlines() if line.strip()]
    for side in [448, 672]:
        data = load(rows, side)
        target = OUT / ('final_%s' % side)
        m = fit(data, sorted(data), winner, 99, target)
        del data
        gc.collect()
        results = []
        for vid in test_ids:
            with np.load(ROOT / ('outputs/spf_ytest_%s' % side) / (vid + '.npz')) as z:
                prof = fused_profile(z['row']) if winner == 'fused_balanced' else z['row'][:, 0].astype(np.float32)
                zs = np.tile(np.array([.5, 0], np.float32), (len(prof), 1))
            c = sh.medfilt(sh.predict(m, prof, zs, 'cuda'))
            assert np.isfinite(c).all() and ((c >= 0) & (c <= 1)).all()
            results.append({'video_id': vid, 'center_y': [round(float(x), 4) for x in c]})
        assert len(results) == len(set(r['video_id'] for r in results)) == 131
        prediction_file = OUT / ('final_%s_test.jsonl' % side)
        prediction_file.write_text(''.join(json.dumps(r) + '\n' for r in results))
        files += [target / 'head.pt', target / 'config.json', prediction_file]
        del m
    write(OUT / 'complete.json', dict(report, complete=True, test_prediction_counts={'448': 131, '672': 131},
          files={str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}))
    print('EXPERIMENT_COMPLETE', winner, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(8)
    rows = [json.loads(line) for line in (ROOT / 'data/ybox_train_v3.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == len({r['video_id'] for r in rows}) == 1516
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(rows)
    else:
        assert not (OUT / 'protocol.json').exists(), 'Do not overwrite a prior experiment'
        run(rows)
