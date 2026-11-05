#!/usr/bin/env python3
"""RetargetVid200 six-reference human-derived 9:16 x-head experiment."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip, medfilt, predict, to_t, crop_iou
from highlight.geometry import safe_crop_size, place_1d
from research_eval import paired

ROOT = Path('outputs/research_44_76/retarget_x')
DATA = Path('outputs/research_44_76/retargetvid_audit/full')
G1_SHA = {448: '1187432d85e03234ef8d807c722ca167063898ce520ff4305bea85c06c5da97f',
          672: 'ad99dc7ed7655d89f991f43b4088f2a4d39cd9041a107b54443a885d0f873064'}


def read(path):
    return [json.loads(line) for line in Path(path).open()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def extent(row):
    return safe_crop_size(row['width'], row['height'], 9, 16)[0] / row['width']


def boxes(centers, row):
    w = safe_crop_size(row['width'], row['height'], 9, 16)[0]
    a = w / row['width']
    c = np.clip(centers, a / 2, 1 - a / 2)
    u = (c * row['width'] - w / 2) / (row['width'] - w)
    return np.array([place_1d(v, row['width'], row['height'], 9, 16) for v in u])


def reference_iou(pred, ref):
    p = pred[:, None, :].astype(np.float64)
    r = ref.astype(np.float64)
    ph = p[..., 2] * 16 / 9
    rh = r[..., 2] * 16 / 9
    iw = np.maximum(0, np.minimum(p[..., 0] + p[..., 2], r[..., 0] + r[..., 2]) - np.maximum(p[..., 0], r[..., 0]))
    ih = np.maximum(0, np.minimum(p[..., 1] + ph, r[..., 1] + rh) - np.maximum(p[..., 1], r[..., 1]))
    intersection = iw * ih
    return intersection / (p[..., 2] * ph + r[..., 2] * rh - intersection)


def load_heads(folder, device, baseline=False):
    result = {}
    for res in (448, 672):
        p = folder / f'head_x_{res}'
        cfg = json.load((p / 'config.json').open())
        if baseline:
            assert sha(p / 'head.pt') == G1_SHA[res]
        assert cfg['axis'] == 'x'
        m = SpatialHead(cfg['D']).to(device).eval()
        m.load_state_dict(torch.load(p / 'head.pt', map_location=device))
        result[res] = m, cfg
    return result


def labels(row):
    return np.load(row['references_path'])


def load_features(row, folder, res, cfg):
    path = folder / f'feats_{res}' / (row['video_id'] + '.npz')
    indices = np.load(path)['frame_indices']
    assert np.array_equal(indices, row['sampling_frame_indices'])
    prof, zs = load_clip(path, 'x', cfg['layer_idx'])
    assert len(prof) == len(indices)
    return prof, zs, indices


def loss_multi(c, target, valid, a):
    legal = c.clamp(a / 2, 1 - a / 2)
    per_reference = 1 - crop_iou(c[:, None], target, a[:, None])
    per_reference += .2 * (legal[:, None] - target).abs() / a[:, None]
    return ((per_reference * valid).sum(1) / valid.sum(1)).mean()


def geometry(rows, out):
    evidence = []
    for row in rows:
        assert row['targetRatioWH'] == [9, 16]
        z = labels(row)
        a = extent(row)
        assert z['centers'].shape == (row['n_frames'], 6)
        assert np.array_equal(z['frame_indices'], np.arange(row['n_frames']))
        assert np.allclose(z['times'], z['frame_indices'] / row['fps'], atol=1e-7, rtol=0)
        sampled = [round(k * row['fps']) for k in range(int(np.floor((row['n_frames'] - 1) / row['fps'])) + 1)]
        assert sampled == row['sampling_frame_indices']
        assert np.all(z['valid'])
        assert np.all(z['centers'] >= a / 2 - 1e-7) and np.all(z['centers'] <= 1 - a / 2 + 1e-7)
        reconstructed = np.stack([boxes(z['centers'][:, i], row) for i in range(6)], axis=1)
        assert np.array_equal(reconstructed, z['derived_9x16_xyw'])
        all_iou = reference_iou(reconstructed[:, 0], reconstructed[:, :1])
        assert np.all(all_iou == 1)
        integer_centers = (reconstructed[..., 0] + reconstructed[..., 2] / 2) / row['width']
        one_d = crop_iou(torch.tensor(integer_centers[:, :1]), torch.tensor(integer_centers),
                         torch.tensor(a, dtype=torch.float64)).numpy()
        assert np.allclose(one_d, reference_iou(reconstructed[:, 0], reconstructed), atol=1e-14, rtol=0)
        first = torch.tensor(z['centers'][[0, -1]], dtype=torch.float64)
        c = first[:, 0]
        v = torch.ones_like(first)
        aa = torch.full_like(c, a)
        check = loss_multi(c, first, v, aa)
        direct = torch.stack([1 - crop_iou(c, first[:, i], aa) + .2 * (c - first[:, i]).abs() / aa
                              for i in range(6)]).mean()
        assert torch.equal(check, direct) or torch.allclose(check, direct, atol=1e-14, rtol=0)
        evidence.append(dict(video_id=row['video_id'], frames=row['n_frames'], references=6,
                             sampled_1fps_frames=len(sampled), original_frame_sampling_exact=True,
                             target='human-derived 9:16 from original 1:3; no center averaging', extent=a,
                             exact_integer_crop_reconstruction=True, self_iou=1., multi_reference_loss_equal=True,
                             continuous_1D_equals_actual_2D_IoU=True,
                             label_sha256=sha(row['references_path'])))
    write(out / 'geometry_check.json', dict(training_started=False, GPU_used=False, sources=evidence))


def train(rows, features, old, out, device, smoke):
    trained = {}
    for res in (448, 672):
        model, cfg = old[res]
        items, buckets = [], {}
        for row in rows:
            prof, zs, indices = load_features(row, features, res, cfg)
            z = labels(row)
            at = np.searchsorted(z['frame_indices'], indices)
            assert np.array_equal(z['frame_indices'][at], indices)
            for i, j in enumerate(at):
                assert z['valid'][j].any()
                item = (prof[i], zs[i], z['centers'][j], extent(row), z['valid'][j])
                buckets.setdefault(len(prof[i]), []).append(len(items))
                items.append(item)
        torch.manual_seed(20261023)
        rng = random.Random(20261023)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.05)
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
                batch = [items[i] for i in ids]
                x, pos, box, target, a = to_t(batch, device)
                valid = torch.tensor(np.stack([b[4] for b in batch]), device=device)
                loss = loss_multi(model(x, pos, box), target, valid, a)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(ids)
            losses.append(total / len(items))
            print('RETARGET_X_EPOCH', res, epoch + 1, losses[-1], flush=True)
        dest = out / f'head_x_{res}'
        dest.mkdir(parents=True, exist_ok=True)
        model.eval()
        torch.save(model.state_dict(), dest / 'head.pt')
        config = dict(cfg, seed=20261023, epochs=len(losses), lr=5e-5, weight_decay=.05, batch=64,
                      warmstart_sha256=G1_SHA[res], loss='mean_reference(1-IoU + .2*legal_center_L1/extent)',
                      sources=len(rows), sampled_frames=len(items), losses=losses, smoke_only=smoke,
                      parameters=sum(p.numel() for p in model.parameters()))
        write(dest / 'config.json', config)
        restored = SpatialHead(cfg['D']).to(device).eval()
        restored.load_state_dict(torch.load(dest / 'head.pt', map_location=device))
        x, p, b, _, _ = to_t(items[:1], device)
        with torch.no_grad():
            assert torch.equal(model(x, p, b), restored(x, p, b))
        trained[res] = model, config
    return trained


@torch.no_grad()
def trajectory(row, features, models, device):
    cs = []
    for res in (448, 672):
        m, cfg = models[res]
        prof, zs, indices = load_features(row, features, res, cfg)
        # G1 per-resolution head files: median then round4.
        cs.append(np.array([round(float(c), 4) for c in medfilt(predict(m, prof, zs, device))]))
    fused = np.array([round(float(c), 4) for c in np.mean(cs, axis=0)])
    # restore_spatial.sh rounds the fusion; ground_build.py applies a second median.
    return indices / row['fps'], medfilt(fused)


@torch.no_grad()
def evaluate(rows, features, baseline, candidate, out, device):
    results, predictions = {}, []
    for row in rows:
        z = labels(row)
        values = {}
        for name, models in [('G1', baseline), ('candidate', candidate)]:
            times, c = trajectory(row, features, models, device)
            a = extent(row)
            legal = np.clip(np.interp(z['times'], times, c), a / 2, 1 - a / 2)
            pred = boxes(legal, row)
            iou = reference_iou(pred, z['derived_9x16_xyw'])
            frame_scores = (iou * z['valid']).sum(1) / z['valid'].sum(1)
            values[name] = float(frame_scores.mean())
            predictions.append(dict(video_id=row['video_id'], model=name, times=z['times'].tolist(), center_x=legal.tolist()))
        results[row['video_id']] = values
    report = dict(metric='six-reference human-derived 9:16 integer 2D IoU, frame mean then source mean',
                  reference='official six-person 1:3 trajectories converted geometrically; not native 9:16 annotations',
                  blind_read=False, per_source=results, splits={})
    for split in sorted({r['dataset_split'] for r in rows}):
        groups = {r['video_id']: r['source_group'] for r in rows if r['dataset_split'] == split}
        report['splits'][split] = paired({v: results[v]['G1'] for v in groups},
                                        {v: results[v]['candidate'] for v in groups}, groups)
    write(out / 'report.json', report)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    print('RETARGET_X_EVAL', json.dumps(report['splits']), flush=True)


@torch.no_grad()
def transfer(base, candidate, out, device):
    root = base.parent
    human_path = root / 'teacher_human_dev_v2/human_dev_frozen.jsonl'
    human = {r['video_id']: r for r in read(human_path)}
    rows = [r for r in read(root / 'human_residual/exact_rows.jsonl')
            if r['width'] / r['height'] > r['targetRatioWH'][0] / r['targetRatioWH'][1]
            and not human[r['video_id']].get('spatial_skipped')]
    assert len(rows) == 15 and sum(sum(not p['ambiguous'] for p in human[r['video_id']]['points']) for r in rows) == 45
    old = load_heads(base, device, baseline=True)
    values = {}
    for row in rows:
        scores = {}
        for name, models in [('G1', old), ('candidate', candidate)]:
            cs = []
            for res in (448, 672):
                m, cfg = models[res]
                path = base / f'feats_{res}' / (row['video_id'] + '.npz')
                prof, zs = load_clip(path, 'x', cfg['layer_idx'])
                # These historical caches have no frame_indices. Preserve their known 0,1,... second time axis.
                assert len(prof) == int(np.floor((row['n_frames'] - 1) / row['fps'])) + 1
                cs.append(np.array([round(float(c), 4) for c in medfilt(predict(m, prof, zs, device))]))
            c = medfilt(np.array([round(float(c), 4) for c in np.mean(cs, axis=0)]))
            points = [p for p in human[row['video_id']]['points'] if not p['ambiguous']]
            a = extent(row)
            truth = np.clip([p['center'] for p in points], a / 2, 1 - a / 2)
            pred = np.clip(np.interp([p['time_sec'] for p in points], np.arange(len(c)), c), a / 2, 1 - a / 2)
            distance = np.abs(pred - truth)
            scores[name] = float(np.mean(np.maximum(0, a - distance) / (a + distance)))
        values[row['video_id']] = scores
    groups = {r['video_id']: r['provenance']['source_group'] for r in rows}
    write(out / 'transfer_report.json', dict(points=45, independent=False, development_previously_used=True,
          blind_read=False, comparison=paired({v: values[v]['G1'] for v in groups},
          {v: values[v]['candidate'] for v in groups}, groups), per_source=values,
          human_sha256=sha(human_path), timing='Historical 1FPS caches lack frame_indices; original integer-second axis retained for both heads.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['geometry', 'train', 'predict', 'eval', 'transfer'], required=True)
    ap.add_argument('--rows', type=Path, default=DATA / 'all_clean.jsonl')
    ap.add_argument('--ready', type=Path, default=DATA / 'READY.json')
    ap.add_argument('--features', type=Path, default=ROOT)
    ap.add_argument('--base', type=Path, default=Path('outputs/research_1022/review_baseline'))
    ap.add_argument('--model', type=Path, default=ROOT / 'train')
    ap.add_argument('--out', type=Path, default=ROOT / 'train')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'geometry':
        rows = read(args.rows)
        assert len(rows) == 2
        geometry(rows, args.out)
        return
    if args.mode == 'transfer':
        transfer(args.base, load_heads(args.model, args.device), args.out, args.device)
        return
    ready = json.load(args.ready.open())
    assert ready['status'] == 'READY' and ready['source_count'] == 200
    assert ready['planned_split_counts'] == dict(train=160, dev=20, heldout=20)
    rows = read(args.rows)
    assert sha(args.rows) == ready['all_clean_sha256']
    assert len(rows) == ready['qualified_source_count'] == len({r['source_group'] for r in rows})
    assert {s: sum(r['dataset_split'] == s for r in rows) for s in ready['split_counts']} == ready['split_counts']
    assert all(sha(r['references_path']) == ready['labels_sha256'][r['video_id']] for r in rows)
    old = load_heads(args.base, args.device, baseline=True)
    if args.mode == 'train':
        tr = [r for r in rows if r['dataset_split'] == 'train']
        val = [r for r in rows if r['dataset_split'] != 'train']
        if args.smoke:
            tr, val = tr[:1], [r for r in val if r['dataset_split'] == 'dev'][:1]
        write(args.out / 'plan.json', dict(seed=20261023, epochs=1 if args.smoke else 8, lr=5e-5,
              optimizer='AdamW', weight_decay=.05, batch=64, frame_weight='equal; six references share one feature',
              loss='mean_valid_reference(1 - IoU + .2 * legal-center L1 / safe crop extent), no center averaging',
              postprocess='per-resolution medfilt(w=1), round4, equal mean round4, medfilt(w=1), interp, legal clamp; exact G1',
              layers={res: cfg['layer_idx'] for res, (_, cfg) in old.items()}, warmstart=G1_SHA,
              train_sources=len(tr), ready_sha256=sha(args.ready), rows_sha256=sha(args.rows),
              script_sha256=sha(__file__), native_9x16_labels=False, blind_read=False, no_hyperparameter_search=True))
        candidate = train(tr, args.features, old, args.out, args.device, args.smoke)
        # Reload the untouched G1 heads: training mutated the warmstart objects.
        old = load_heads(args.base, args.device, baseline=True)
        evaluate(val, args.features, old, candidate, args.out, args.device)
    elif args.mode == 'eval':
        evaluate([r for r in rows if r['dataset_split'] != 'train'], args.features, old,
                 load_heads(args.model, args.device), args.out, args.device)
    else:
        models = load_heads(args.model, args.device)
        with (args.out / 'predictions.jsonl').open('w') as f:
            for row in rows:
                try:
                    t, c = trajectory(row, args.features, models, args.device)
                    full_times = np.arange(row['n_frames']) / row['fps']
                    a = extent(row)
                    legal = np.clip(np.interp(full_times, t, c), a / 2, 1 - a / 2)
                    record = dict(video_id=row['video_id'], times=full_times.tolist(), center_x=legal.tolist())
                except Exception as e:
                    print('error', row['video_id'], repr(e), flush=True)
                    record = dict(video_id=row['video_id'], predictions=[])
                f.write(json.dumps(record) + '\n')
    write(args.out / 'manifest.json', {str(p): sha(p) for p in args.out.rglob('*') if p.is_file() and p.name != 'manifest.json'})


if __name__ == '__main__':
    main()
