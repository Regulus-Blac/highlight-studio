#!/usr/bin/env python3
"""Full16:9 CPU prototype: frozen X1/G1 marginals, continuous legal2D crop residuals."""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip
from reframe import geometry
from research_eval import paired
from highlight.geometry import center_crop, dof_range, place_1d, safe_crop_size


SEED = 20261023


class CropHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).mean(1)  # identical head, equal 448/672 residual mean


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_heads(g1, x1):
    models, inputs = {}, {}
    for res in (448, 672):
        for axis, root in [('x', x1), ('y', g1)]:
            folder = root / f'head_{axis}_{res}'
            cfg = json.loads((folder / 'config.json').read_text())
            assert cfg['axis'] == axis
            model = SpatialHead(cfg['D']).eval().requires_grad_(False)
            model.load_state_dict(torch.load(folder / 'head.pt', map_location='cpu', weights_only=True))
            models[res, axis] = model, cfg
            inputs.update({str(p): sha(p) for p in [folder / 'head.pt', folder / 'config.json']})
    return models, inputs


@torch.no_grad()
def encode(model, profile, zero_shot):
    x = torch.from_numpy(profile.astype(np.float32))
    pos = (torch.arange(x.shape[1], dtype=torch.float32) + .5) / x.shape[1]
    pos = pos[None].expand(len(x), -1)
    box = torch.from_numpy(zero_shot)
    hidden = model.enc(model.proj(x) + model.pos(pos[..., None]) + model.box(box)[:, None])
    prob = model.out(hidden).squeeze(-1).softmax(-1)
    center = (prob * pos).sum(-1)
    assert torch.equal(center, model(x, pos, box))
    u = 2 * pos - 1
    moments = [hidden.mean(1), (hidden * u[..., None]).mean(1), (hidden * u.square()[..., None]).mean(1)]
    stats = torch.stack([center, (prob * (pos - center[:, None]).square()).sum(-1).sqrt(),
                         -(prob * prob.clamp(min=1e-12).log()).sum(-1) / math.log(x.shape[1]),
                         prob.max(-1).values], -1)
    return torch.cat([*moments, stats], -1).numpy(), center.numpy()


def baseline_box(row, centers):
    """One image/anchor: X1 x rounding, G1 y raw mean; full deliberately uses maximum size."""
    width, height = row['width'], row['height']
    tw, th = row['targetRatioWH']
    w, _ = safe_crop_size(width, height, tw, th)
    g = geometry(width, height, [tw, th])
    if g is None:
        return np.array(center_crop(width, height, tw, th), np.int64)
    if g[0] == 'x':
        c = round(sum(round(float(centers[res, 'x']), 4) for res in (448, 672)) / 2, 4)
        return np.array(place_1d((c * width - w / 2) / dof_range(width, height, tw, th),
                                width, height, tw, th), np.int64)
    h = w * th / tw
    c = np.clip(np.mean([centers[res, 'y'] for res in (448, 672)]), h / height / 2, 1 - h / height / 2)
    return np.array([0, min(max(0, round(float(c) * height - h / 2)), int(height - h)), w], np.int64)


def continuous_box(residual, base):
    scale = residual[:, 2].exp().clamp(max=1.)
    size = base[:, 2:] * scale[:, None]
    origin = base[:, :2] + (base[:, 2:] - size) / 2 + residual[:, :2].tanh()
    origin = torch.minimum(origin.clamp(min=0), 1 - size)
    return torch.cat([origin, origin + size], -1)


def integer_box(residual, row, base):
    width, height = row['width'], row['height']
    tw, th = row['targetRatioWH']
    x, y, old_w = map(int, base)
    old_h = old_w * th / tw
    w = max(1, int(old_w * math.exp(min(0., float(residual[2])))))
    h = w * th / tw
    x = min(max(0, round(x + (old_w - w) / 2 + math.tanh(float(residual[0])) * width)), width - w)
    y = min(max(0, round(y + (old_h - h) / 2 + math.tanh(float(residual[1])) * height)), int(height - h))
    return np.array([x, y, w], np.int64)


def iou(pred, target):
    intersection = (torch.minimum(pred[:, 2:], target[:, 2:]) -
                    torch.maximum(pred[:, :2], target[:, :2])).clamp(min=0).prod(-1)
    return intersection / ((pred[:, 2:] - pred[:, :2]).prod(-1) +
                           (target[:, 2:] - target[:, :2]).prod(-1) - intersection)


def normalized_box(box, row):
    x, y, w = map(float, box)
    h = w * row['targetRatioWH'][1] / row['targetRatioWH'][0]
    return np.array([x / row['width'], y / row['height'], w / row['width'], h / row['height']], np.float32)


def geo_features(row, base):
    b = normalized_box(base, row)
    return np.array([math.log(row['width'] / row['height']),
                     math.log(row['targetRatioWH'][0] / row['targetRatioWH'][1]),
                     *b[2:], *(b[:2] + b[2:] / 2)], np.float32)


def original_iou(box, row):
    b = normalized_box(box, row).astype(np.float64)
    p = torch.tensor([np.r_[b[:2], b[:2] + b[2:]]], dtype=torch.float64)
    return float(iou(p, torch.tensor([row['original_crop_xyxy']], dtype=torch.float64))[0])


def prepare(rows, features, models, out, split):
    vectors, bases, triples, inputs = [], [], [], {}
    for number, row in enumerate(rows, 1):
        assert row['targetRatioWH'] == [16, 9], 'Only real16:9 supervision is qualified for this prototype'
        encoded, centers = {}, {}
        for res in (448, 672):
            path = features / f'feats_{res}' / (row['video_id'] + '.npz')
            stored = np.load(path)
            assert len(stored['frame_indices']) == 1  # original GNMC image, no temporal labels
            for axis in ('x', 'y'):
                model, cfg = models[res, axis]
                profile, zs = load_clip(path, axis, cfg['layer_idx'])
                encoded[res, axis], center = encode(model, profile, zs)
                centers[res, axis] = float(center[0])
            inputs[str(path)] = sha(path)
        box = baseline_box(row, centers)
        base = normalized_box(box, row)
        geo = geo_features(row, box)
        vectors.append(np.stack([np.r_[encoded[res, 'x'][0], encoded[res, 'y'][0], geo] for res in (448, 672)]))
        bases.append(base)
        triples.append(box)
        assert np.array_equal(integer_box(np.zeros(3), row, box), box)
        if number % 100 == 0:
            print('CROP_SCALE_PREPARE', split, number, len(rows), flush=True)
    np.savez(out / (split + '.npz'), features=np.asarray(vectors, np.float32), base=np.asarray(bases, np.float32),
             triplets=np.asarray(triples), target=np.array([r['original_crop_xyxy'] for r in rows], np.float32))
    (out / (split + '_rows.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / (split + '_features_sha256.json')).write_text(json.dumps(inputs, indent=2) + '\n')


def train(out, smoke=False):
    data = np.load(out / ('smoke.npz' if smoke else 'train.npz'))
    features, base, target = [torch.from_numpy(data[k]) for k in ('features', 'base', 'target')]
    torch.manual_seed(SEED)
    model = CropHead(features.shape[-1])
    with torch.no_grad():
        assert torch.equal(model(features[:2]), torch.zeros(len(features[:2]), 3))
        initial = continuous_box(model(features[:2]), base[:2])
        assert torch.equal(initial, torch.cat([base[:2, :2], base[:2, :2] + base[:2, 2:]], -1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.05)
    rng, losses = random.Random(SEED), []
    for epoch in range(1 if smoke else 8):
        ids = list(range(len(features)))
        rng.shuffle(ids)
        total = 0.
        for start in range(0, len(ids), 64):
            batch = ids[start:start + 64]
            pred = continuous_box(model(features[batch]), base[batch])
            pc, tc = (pred[:, :2] + pred[:, 2:]) / 2, (target[batch, :2] + target[batch, 2:]) / 2
            scale = (pred[:, 2] - pred[:, 0]) / base[batch, 2]
            target_scale = (((target[batch, 2:] - target[batch, :2]).prod(-1) /
                             base[batch, 2:].prod(-1)).sqrt()).clamp(max=1.)
            loss = (1 - iou(pred, target[batch]) + .2 * ((pc - tc).abs() / base[batch, 2:]).mean(-1) +
                    .2 * (scale - target_scale).abs()).mean()
            optimizer.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(features))
        print('CROP_SCALE_EPOCH', epoch + 1, losses[-1], flush=True)
    model.eval()
    stem = 'smoke_head' if smoke else 'head'
    torch.save(model.state_dict(), out / (stem + '.pt'))
    cfg = dict(dimension=features.shape[-1], parameters=sum(p.numel() for p in model.parameters()),
               seed=SEED, epochs=len(losses), lr=5e-5, weight_decay=.05, batch=64, losses=losses,
               loss='1-original2D IoU + .2 legal-center L1/max extent + .2 scale L1',
               target_scale='sqrt(original human area / maximum legal crop area), capped at1; original IoU target unchanged',
               zero_init_function_equal=True, full_initialization='maximum legal box, not X1 full92', smoke_only=smoke)
    (out / (stem + '.json')).write_text(json.dumps(cfg, indent=2) + '\n')
    restored = CropHead(cfg['dimension']).eval()
    restored.load_state_dict(torch.load(out / (stem + '.pt'), map_location='cpu', weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(features[:2]), restored(features[:2]))
    return model, cfg


@torch.no_grad()
def evaluate(out):
    cfg = json.loads((out / 'head.json').read_text())
    model = CropHead(cfg['dimension']).eval()
    model.load_state_dict(torch.load(out / 'head.pt', map_location='cpu', weights_only=True))
    data = np.load(out / 'validation.npz')
    rows = list(map(json.loads, (out / 'validation_rows.jsonl').open()))
    residuals = model(torch.from_numpy(data['features'])).numpy()
    values, predictions = {}, []
    for row, base, residual in zip(rows, data['triplets'], residuals):
        kind = geometry(row['width'], row['height'], row['targetRatioWH'])
        kind = 'full' if kind is None else kind[0]
        box = integer_box(residual, row, base)
        scale_only = integer_box([0., 0., residual[2]], row, base)
        reference = base.copy()
        if kind == 'full':  # diagnostic only; not a feature, init, label or training target
            w = int(base[2] * .92)
            h = w * 9 / 16
            reference = np.array([min(max(0, round((row['width'] - w) / 2)), row['width'] - w),
                                  min(max(0, round((row['height'] - h) / 2)), int(row['height'] - h)), w])
        scores = dict(maximum=original_iou(base, row), X1_reference=original_iou(reference, row),
                      scale_only_readout=original_iou(scale_only, row), learned=original_iou(box, row))
        values[row['video_id']] = scores
        predictions.append(dict(video_id=row['video_id'], geometry=kind, box_xyw=box.tolist(),
                                width_scale=float(box[2] / base[2]), scores=scores))
    report = dict(training_policy='fixed8 epochs, no validation selection', independent_validation=False,
                  no_test_or_blind=True, online_baseline='X1 user-reported43.30; no local-to-online score conversion',
                  primary='joint center+scale; scale-only is a fixed-center readout of the same trained model, not a separately trained control or selectable fallback',
                  production_scope='16:9 full geometry only; all nonfull rows must remain X1 unchanged',
                  parameters=cfg['parameters'], full_pipeline_parameters=8926500280 + cfg['parameters'], subsets={})
    for subset in ('all', 'full', 'x', 'y'):
        chosen = [r for r, p in zip(rows, predictions) if subset == 'all' or p['geometry'] == subset]
        if not chosen:
            continue
        groups = {r['video_id']: r['provenance']['source_group'] for r in chosen}
        report['subsets'][subset] = dict(images=len(chosen), sources=len(set(groups.values())),
            means={name: float(np.mean([values[v][name] for v in groups])) for name in ('maximum', 'X1_reference', 'scale_only_readout', 'learned')},
            paired={name: paired({v: values[v][name] for v in groups}, {v: values[v]['learned'] for v in groups}, groups)
                    for name in ('maximum', 'X1_reference', 'scale_only_readout')})
        report['subsets'][subset]['effect_distribution'] = {}
        for name in ('maximum', 'X1_reference', 'scale_only_readout'):
            delta = np.array([values[v]['learned'] - values[v][name] for v in groups])
            report['subsets'][subset]['effect_distribution'][name] = dict(
                quantiles_05_25_50_75_95=np.quantile(delta, [.05, .25, .5, .75, .95]).tolist(),
                improved=int((delta > 0).sum()), tied=int((delta == 0).sum()), regressed=int((delta < 0).sum()),
                gain_at_least_05=int((delta >= .05).sum()), loss_at_least_05=int((delta <= -.05).sum()))
    (out / 'validation_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CROP_SCALE_EVALUATION', json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prepare', 'train', 'evaluate', 'smoke'], required=True)
    ap.add_argument('--data', type=Path, default=Path('outputs/research_44_76/learned_box/full'))
    ap.add_argument('--features', type=Path, default=Path('outputs/research_44_76/learned_box/features'))
    ap.add_argument('--g1', type=Path, default=Path('outputs/research_44_76/gnmc_train'))
    ap.add_argument('--x1', type=Path, default=Path('outputs/research_44_76/retarget_x/train'))
    ap.add_argument('--out', type=Path, default=Path('outputs/research_44_76/crop_scale'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in ('prepare', 'smoke'):
        models, inputs = load_heads(args.g1, args.x1)
        if args.mode == 'smoke':
            rows = list(map(json.loads, (args.data / 'train_clean.jsonl').open()))
            rows = [r for r in rows if geometry(r['width'], r['height'], r['targetRatioWH']) is None][:2]
            assert len(rows) == 2, 'Two real public full-geometry training images are required'
            prepare(rows, args.features, models, args.out, 'smoke')
            model, cfg = train(args.out, smoke=True)
            print('CROP_SCALE_CPU_SMOKE_COMPLETE', json.dumps(cfg), flush=True)
        else:
            train_rows = list(map(json.loads, (args.data / 'train_clean.jsonl').open()))
            val_rows = list(map(json.loads, (args.data / 'validation_clean.jsonl').open()))
            assert not {r['provenance']['source_group'] for r in train_rows} & {r['provenance']['source_group'] for r in val_rows}
            for split, rows in [('train', train_rows), ('validation', val_rows)]:
                rows = [r for r in rows if geometry(r['width'], r['height'], r['targetRatioWH']) is None]
                assert rows, 'Existing y-only caches cannot qualify a full-geometry model'
                prepare(rows, args.features, models, args.out, split)
        inputs.update({str(p): sha(p) for p in [args.data / 'train_clean.jsonl', args.data / 'validation_clean.jsonl']
                       if p.exists()})
        helpers = [Path(__file__), Path('tools/train_spatial_head.py'), Path('tools/reframe.py'),
                   Path('highlight/geometry.py')]
        (args.out / 'inputs_sha256.json').write_text(json.dumps({**{str(p): sha(p) for p in helpers}, **inputs}, indent=2) + '\n')
    elif args.mode == 'train':
        train(args.out)
    else:
        evaluate(args.out)


if __name__ == '__main__':
    main()
