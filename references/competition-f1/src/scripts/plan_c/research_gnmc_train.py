#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip, predict, to_t
from research_residual_features import readout
from research_eval import paired


def extent(row):
    tw, th = row['targetRatioWH']
    return int(min(row['width'], row['height'] * tw / th)) * th / tw / row['height']


def crop_iou(center, boxes, a):
    top = center.clamp(a / 2, 1 - a / 2) - a / 2
    overlap = (torch.minimum(top + a, boxes[:, 3]) - torch.maximum(top, boxes[:, 1])).clamp(min=0)
    width = boxes[:, 2] - boxes[:, 0]
    intersection = overlap * width
    return intersection / (a + width * (boxes[:, 3] - boxes[:, 1]) - intersection)


def score(center, row, integer=True):
    a = extent(row)
    c = float(np.clip(center, a / 2, 1 - a / 2))
    if integer:
        h = a * row['height']
        y = min(max(0, round(c * row['height'] - h / 2)), int(row['height'] - h))
        c = (y + h / 2) / row['height']
    return float(crop_iou(torch.tensor([c], dtype=torch.float64),
                         torch.tensor([row['original_crop_xyxy']], dtype=torch.float64),
                         torch.tensor([a], dtype=torch.float64))[0])


def load_head(base, res, device):
    folder = base / 'review_baseline' / f'head_y_{res}'
    cfg = json.loads((folder / 'config.json').read_text())
    model = SpatialHead(cfg['D']).to(device)
    model.load_state_dict(torch.load(folder / 'head.pt', map_location=device))
    return model.eval(), cfg


def train(rows, features, base, out, res, device, smoke):
    model, cfg = load_head(base, res, device)
    items, buckets = [], {}
    for row in rows:
        prof, zs = load_clip(features / f'feats_{res}' / (row['video_id'] + '.npz'), 'y', cfg['layer_idx'])
        assert len(prof) == 1
        item = (prof[0], zs[0], row['teacher_signals']['trajectory'][0]['center'][1], extent(row))
        buckets.setdefault(len(prof[0]), []).append(len(items))
        items.append(item)
    torch.manual_seed(20261022)
    rng = random.Random(20261022)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.05)
    losses = []
    for epoch in range(1 if smoke else 8):
        model.train()
        batches = []
        for group in buckets.values():
            ids = list(group)
            rng.shuffle(ids)
            batches.extend(ids[i:i + 64] for i in range(0, len(ids), 64))
        rng.shuffle(batches)
        total = 0.
        for ids in batches:
            x, pos, box, target, a = to_t([items[i] for i in ids], device)
            gt = torch.tensor([rows[i]['original_crop_xyxy'] for i in ids], dtype=torch.float32, device=device)
            c = model(x, pos, box)
            legal = c.clamp(a / 2, 1 - a / 2)
            loss = (1 - crop_iou(c, gt, a) + .2 * (legal - target).abs() / a).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(items))
        print('GNMC EPOCH', res, epoch + 1, losses[-1], flush=True)
    dest = out / f'head_y_{res}'
    dest.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), dest / 'head.pt')
    config = {'D': cfg['D'], 'axis': 'y', 'layer_idx': cfg['layer_idx'], 'seed': 20261022,
              'epochs': len(losses), 'lr': 5e-5, 'batch': 64, 'weight_decay': .05,
              'loss': '1 - original-human-2D-IoU + .2 * legal-center-L1 / max-crop-extent',
              'training_rows': len(rows), 'training_ids': [r['video_id'] for r in rows],
              'losses': losses, 'grid_cell_buckets': {str(k): len(v) for k, v in buckets.items()},
              'parameters': sum(p.numel() for p in model.parameters()), 'smoke_only': smoke}
    (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    restored = SpatialHead(cfg['D']).to(device).eval()
    restored.load_state_dict(torch.load(dest / 'head.pt', map_location=device))
    x, p, b, _, _ = to_t([items[0]], device)
    with torch.no_grad():
        assert torch.equal(model(x, p, b), restored(x, p, b))
    return model, cfg


def summarize(rows, values, reference, names):
    result = {}
    for subset, keep in [('all', rows), ('portrait', [r for r in rows if r['portrait']]),
                         ('landscape', [r for r in rows if not r['portrait']])]:
        if not keep:
            continue
        groups = {r['video_id']: r['provenance']['source_group'] for r in keep}
        result[subset] = {'images': len(keep), 'means': {name: float(np.mean([values[r['video_id']][name] for r in keep]))
                                                     for name in values[keep[0]['video_id']]},
                          'paired': {name: paired({v: values[v][reference] for v in groups},
                                                  {v: values[v][name] for v in groups}, groups) for name in names}}
    return result


@torch.no_grad()
def validation(rows, features, base, models, out, device):
    old = {res: load_head(base, res, device) for res in (448, 672)}
    residual = np.load(base / 'human_residual/selected_model.npz')
    values, predictions = {}, []
    for row in rows:
        v = row['video_id']
        cs, stats, new = [], [], []
        for res in (448, 672):
            model, cfg = old[res]
            prof, zs = load_clip(features / f'feats_{res}' / (v + '.npz'), 'y', cfg['layer_idx'])
            # H1 readout is CPU-only; keep this fixed baseline separate from training.
            model = model.to('cpu')
            _, s = readout(model, prof, zs)
            cs.append(s[:, 0])
            stats.append(s)
            m, cfg = models[res]
            new.append(predict(m, prof, zs, device))
        raw = float(np.mean(cs))
        candidate = float(np.mean(new))
        a = extent(row)
        geo = np.array([cs[0][0], cs[1][0], *[s[0, i] for s in stats for i in range(4)], 0.])
        delta = np.clip(geo @ residual['coef'] + residual['bias'], -residual['bound'], residual['bound'])
        h1 = np.clip(raw, a / 2, 1 - a / 2) + residual['base_shift'] + delta
        centers = {'R43': raw, 'R43_shifted': np.clip(raw, a / 2, 1 - a / 2) + .015,
                   'H1_fixed': float(h1), 'gnmc_raw': candidate, 'gnmc_half': (candidate + raw) / 2}
        values[v] = {name: score(c, row) for name, c in centers.items()}
        values[v]['maximum_width_oracle_continuous'] = score(row['max_size_center_y'], row, integer=False)
        predictions.append({'video_id': v, 'centers': centers, 'iou': values[v]})
    (out / 'validation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    report = summarize(rows, values, 'H1_fixed', ['gnmc_raw', 'gnmc_half'])
    report['vs_R43'] = summarize(rows, values, 'R43', ['gnmc_raw', 'gnmc_half'])
    report['notes'] = ['Original human 2D xyxy is unchanged; predicted max-width crops use integer y as in submissions.',
                       'Oracle is the continuous maximum-width projection, an upper bound on this 1D prediction family.',
                       'Official GNMC validation is not used for early stopping, epoch/lr choice or mixing weights.',
                       'GNMC source identities are pixel hashes; content isolation does not prove unseen author/source identity.',
                       'H1 is fixed on its original R43 inputs; its residual is never applied to the new heads.']
    (out / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GNMC VALIDATION', json.dumps({k: v for k, v in report.items() if k != 'notes'}), flush=True)


@torch.no_grad()
def transfer(base, models, out, device):
    folder = base / 'human_residual/dense'
    human = {r['video_id']: r for r in map(json.loads, (base / 'teacher_human_dev_v2/human_dev_frozen.jsonl').open())}
    rows = [r for r in map(json.loads, (folder / 'rows.jsonl').open()) if not human[r['video_id']].get('spatial_skipped')]
    h1 = {r['video_id']: r for r in map(json.loads, (folder / 'train/y_geometry_oof.jsonl').open())}
    old = {res: load_head(base, res, device) for res in (448, 672)}
    values, predictions = {}, []
    for row in rows:
        v = row['video_id']
        raw, new = [], []
        for res in (448, 672):
            m, cfg = old[res]
            prof, zs = load_clip(folder / f'feats_{res}' / (v + '.npz'), 'y', cfg['layer_idx'])
            raw.append(predict(m, prof, zs, device))
            new.append(predict(models[res][0], prof, zs, device))
        raw, new = np.mean(raw, axis=0), np.mean(new, axis=0)
        times = np.array(row['sampling_frame_indices']) / row['fps']
        a = extent(row)
        candidates = {'gnmc_raw': np.clip(new, a / 2, 1 - a / 2),
                      'gnmc_half': np.clip((new + raw) / 2, a / 2, 1 - a / 2)}
        series = {'R43_raw': (times, raw), 'H1_oof': (np.array(h1[v]['times']), np.array(h1[v]['center_y'])),
                  **{name: (times, c) for name, c in candidates.items()}}
        scores = {name: [] for name in series}
        for p in human[v]['points']:
            if p['ambiguous']:
                continue
            target = np.clip(p['center'], a / 2, 1 - a / 2)
            for name, (t, c) in series.items():
                distance = abs(np.clip(np.interp(p['time_sec'], t, c), a / 2, 1 - a / 2) - target)
                scores[name].append(max(0., a - distance) / (a + distance))
        values[v] = {name: float(np.mean(s)) for name, s in scores.items()}
        predictions.append({'video_id': v, 'times': times.tolist(), **{name: c.tolist() for name, c in candidates.items()}})
    groups = {r['video_id']: r['provenance']['source_group'] for r in rows}
    report = {'independent_validation': False, 'blind_read': False, 'points': 54,
              'comparison': {name: paired({v: values[v]['H1_oof'] for v in groups},
                                         {v: values[v][name] for v in groups}, groups) for name in candidates},
              'per_video': values, 'notes': ['These human development labels trained H1 previously; this is an exploratory transfer check.',
                                           'GNMC training does not consume these human answers. H1 uses its frozen source-held-out OOF predictions.',
                                           'Both new raw and fixed half-R43 blend were declared before this evaluation; no H1 correction is added.']}
    (out / 'transfer_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    (out / 'transfer_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GNMC TRANSFER', json.dumps(report['comparison']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='outputs/research_44_76/gnmc_y')
    ap.add_argument('--features', default='outputs/research_44_76/gnmc_y')
    ap.add_argument('--base', default='outputs/research_1022')
    ap.add_argument('--out', default='outputs/research_44_76/gnmc_train')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--transfer-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    data, features, base, out = map(Path, (args.data, args.features, args.base, args.out))
    out.mkdir(parents=True, exist_ok=True)
    if args.transfer_only:
        models = {}
        for res in (448, 672):
            folder = out / f'head_y_{res}'
            cfg = json.loads((folder / 'config.json').read_text())
            m = SpatialHead(cfg['D']).to(args.device).eval()
            m.load_state_dict(torch.load(folder / 'head.pt', map_location=args.device))
            models[res] = m, cfg
        transfer(base, models, out, args.device)
        paths = [Path(__file__), out / 'transfer_report.json', base / 'teacher_human_dev_v2/human_dev_frozen.jsonl']
        paths += [out / f'head_y_{res}/head.pt' for res in (448, 672)]
        (out / 'transfer_manifest.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                               for p in paths}, indent=2) + '\n')
        print('GNMC TRANSFER COMPLETE', flush=True)
        return
    train_file, val_file = data / 'train_clean.jsonl', data / 'validation_clean.jsonl'
    rows = list(map(json.loads, train_file.open()))
    val = list(map(json.loads, val_file.open()))
    assert all(r['dataset_split'] == 'train' for r in rows)
    assert all(r['dataset_split'] == 'validation' for r in val)
    assert not {r['provenance']['source_group'] for r in rows} & {r['provenance']['source_group'] for r in val}
    if args.smoke:
        rows, val = rows[:1], val[:1]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs = [Path(__file__), train_file, val_file, base / 'human_residual/selected_model.npz']
    inputs += [base / 'review_baseline' / f'head_y_{res}/head.pt' for res in (448, 672)]
    plan = {'warmstart': 'all R43 y SpatialHead parameters', 'epochs': 1 if args.smoke else 8,
            'lr': 5e-5, 'batch': 64, 'weight_decay': .05, 'seed': 20261022, 'image_weight': 'equal',
            'batch_policy': 'same grid-cell-count buckets, shuffled each epoch; no repeated-cell padding across image shapes',
            'loss': 'true original-human-2D IoU plus .2 legal-center L1 / max-width crop extent',
            'train': len(rows), 'validation': len(val), 'smoke_only': args.smoke,
            'candidate_policy': 'primary raw new-head mean; second fixed 50/50 with R43 raw mean; no +.015 or H1 residual on either',
            'H1_policy': 'fixed original R43+human residual control on GNMC; frozen H1 source-OOF reference for public dev transfer',
            'no_val_tuning': True, 'blind_read': False, 'test_data_used_for_training': False,
            'input_sha256': {str(p): sha(p) for p in inputs}}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    started = time.time()
    models = {res: train(rows, features, base, out, res, args.device, args.smoke) for res in (448, 672)}
    validation(val, features, base, models, out, args.device)
    result = {'seconds': time.time() - started, 'device': args.device, 'smoke_only': args.smoke,
              'parameters_per_head': {str(res): sum(p.numel() for p in model.parameters()) for res, (model, _) in models.items()},
              'input_sha256': plan['input_sha256'],
              'output_sha256': {str(p): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('GNMC TRAIN COMPLETE', json.dumps({k: v for k, v in result.items() if 'sha256' not in k}), flush=True)


if __name__ == '__main__':
    main()
