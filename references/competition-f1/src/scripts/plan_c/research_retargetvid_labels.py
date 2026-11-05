#!/usr/bin/env python3
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np


ROOT = Path('outputs/research_44_76/retargetvid_audit')
OUT = ROOT / 'full'
SEED = 20261023
IDS = [f'{n:03d}' for n in list(range(1, 101)) + list(range(601, 701))]
RATIOS = {'1-3': (1, 3), '3-1': (3, 1)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shuffled = sorted(IDS)
    random.Random(SEED).shuffle(shuffled)
    groups = {'train': shuffled[:160], 'dev': shuffled[160:180], 'heldout': shuffled[180:]}
    splits = {vid: split for split, vids in groups.items() for vid in vids}
    script_sha = sha(Path(__file__).read_bytes())
    inputs, unexpected, sources = [], [], {}
    for vid in IDS:
        sources[vid] = {'source_id': vid, 'source_group': f'DHF1K_{vid}',
                        'official_DHF1K_split': 'train' if int(vid) <= 100 else 'validation',
                        'proposed_split': splits[vid], 'trajectories': [], 'errors': []}

    for annotator in range(1, 7):
        archive_path = ROOT / 'source' / f'annotator_{annotator}.zip'
        inputs.append({'path': str(archive_path), 'bytes': archive_path.stat().st_size,
                       'sha256': sha(archive_path.read_bytes())})
        folder = OUT / 'annotations' / f'annotator_{annotator}'
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            expected = {f'annotator_{annotator}/{vid}_{ratio}.txt'
                        for vid in IDS for ratio in RATIOS}
            unexpected += [{'archive': str(archive_path), 'member': name}
                           for name in archive.namelist() if name.endswith('.txt') and name not in expected]
            for vid in IDS:
                for ratio, (rw, rh) in RATIOS.items():
                    name = f'{vid}_{ratio}.txt'
                    member = f'annotator_{annotator}/' + name
                    row = {'annotator': annotator, 'ratio': ratio, 'archive_member': member,
                           'path': str(folder / name), 'errors': []}
                    sources[vid]['trajectories'].append(row)
                    try:
                        raw = archive.read(member)
                        (folder / name).write_bytes(raw)
                        row.update({'bytes': len(raw), 'sha256': sha(raw)})
                        boxes = np.loadtxt(io.BytesIO(raw), delimiter=',', ndmin=2)
                        row['n_frames'] = len(boxes)
                        if boxes.shape[1] != 4 or len(boxes) == 0:
                            raise ValueError(f'expected nonempty N x 4, got {boxes.shape}')
                        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                        checks = {'nonfinite': ~np.isfinite(boxes).all(axis=1),
                                  'noninteger': (boxes != np.round(boxes)).any(axis=1),
                                  'negative_coordinate': (boxes < 0).any(axis=1),
                                  'out_of_bounds': (boxes[:, 0] > 640) | (boxes[:, 2] > 640)
                                                   | (boxes[:, 1] > 360) | (boxes[:, 3] > 360),
                                  'nonpositive_extent': (w <= 0) | (h <= 0),
                                  'ratio_error_over_one_pixel': np.abs(w * rh - h * rw) > max(rw, rh)}
                        for error, mask in checks.items():
                            if mask.any():
                                frames = np.flatnonzero(mask)
                                row['errors'].append({'kind': error, 'count': len(frames),
                                                      'first_frames': frames[:10].tolist()})
                        positive = h > 0
                        row.update({'crop_sizes_wh': np.unique(np.stack([w, h], axis=1), axis=0).tolist(),
                                    'coordinate_min': boxes.min(axis=0).tolist(),
                                    'coordinate_max': boxes.max(axis=0).tolist(),
                                    'max_ratio_absolute_error': float(np.max(np.abs(w[positive] / h[positive] - rw / rh))),
                                    'max_ratio_crossproduct_error': float(np.max(np.abs(w * rh - h * rw)))})
                    except (KeyError, ValueError, OSError) as error:
                        row['errors'].append({'kind': 'read_or_parse', 'message': str(error)})
                    if row['errors']:
                        sources[vid]['errors'].append({'annotator': annotator, 'ratio': ratio,
                                                       'errors': row['errors']})
                        print('ERROR', vid, annotator, ratio, json.dumps(row['errors']))

    for source in sources.values():
        lengths = sorted({row['n_frames'] for row in source['trajectories'] if 'n_frames' in row})
        source['frame_lengths'] = lengths
        source['all_six_annotators_both_ratios_same_length'] = len(lengths) == 1 and all(
            'n_frames' in row for row in source['trajectories'])
        if not source['all_six_annotators_both_ratios_same_length']:
            source['errors'].append({'kind': 'source_frame_count_mismatch', 'lengths': lengths})
        source['annotation_status'] = 'pending' if source['errors'] else 'passed'
        source['n_frames'] = lengths[0] if source['all_six_annotators_both_ratios_same_length'] else None
        source['training_ready'] = False

    counts = {}
    for split, vids in groups.items():
        ss = [sources[vid] for vid in vids]
        trajectories = [r for s in ss for r in s['trajectories']]
        counts[split] = {'sources': len(ss), 'annotation_passed_sources': sum(not s['errors'] for s in ss),
                         'pending_sources': sum(bool(s['errors']) for s in ss),
                         'trajectories': len(trajectories),
                         'extracted_txt': sum('sha256' in r for r in trajectories),
                         'source_frames_with_consistent_labels': sum(s['n_frames'] or 0 for s in ss),
                         'frame_annotations': sum(r.get('n_frames', 0) for r in trajectories)}
    txt_hashes = {row['path']: row['sha256'] for s in sources.values()
                  for row in s['trajectories'] if 'sha256' in row}
    notes = ['仅审计作者六个原始ZIP，不读取旧144源派生数据；未解码或查看任何媒体。',
             'xyxy为作者零基坐标；以640×360为标签坐标系，边界允许x2=640、y2=360。',
             '比例容差为任一边至多1像素整数舍入：abs(w*rh-h*rw)<=max(rw,rh)，同时记录精确误差。',
             '不修正负值、越界、比例或长度异常；任一标签错误使整个source为pending。',
             '所有六人及两比例按同一DHF1K来源分组；固定划分先于标签审计结果，排重后不补数。',
             '本划分为拟定来源级划分，尚待官方媒体身份核验、帧数匹配及AIC排重；全部training_ready=false。',
             '没有训练或评价模型；没有打开项目测试或已消费盲测；heldout不能用于后续调参。']
    plan = {'seed': SEED, 'algorithm': 'random.Random(20261023).shuffle(sorted(ids)); first160 train, next20 dev, last20 heldout',
            'source_ids': IDS, 'source_ids_sha256': sha(('\n'.join(IDS) + '\n').encode()),
            'shuffle_order': shuffled, 'splits': groups, 'source_to_split': splits,
            'source_to_split_sha256': sha(json.dumps(splits, sort_keys=True, separators=(',', ':')).encode()),
            'actual_counts': counts, 'inputs': inputs, 'script_sha256': script_sha, 'notes': notes}
    write_json(OUT / 'split_plan.json', plan)
    report = {'coordinate_system': {'width': 640, 'height': 360, 'format': 'zero_based_xyxy'},
              'expected_sources': 200, 'expected_trajectories': 2400, 'source_count': len(sources),
              'extracted_txt_count': len(txt_hashes),
              'annotation_passed_sources': sum(not s['errors'] for s in sources.values()),
              'pending_source_ids': [vid for vid, s in sources.items() if s['errors']],
              'source_frames_with_consistent_labels': sum(s['n_frames'] or 0 for s in sources.values()),
              'frame_annotations': sum(r.get('n_frames', 0) for s in sources.values() for r in s['trajectories']),
              'actual_split_counts': counts, 'unexpected_txt_members': unexpected,
              'inputs': inputs, 'script_sha256': script_sha,
              'raw_txt_sha256': txt_hashes,
              'raw_txt_hash_manifest_sha256': sha(json.dumps(txt_hashes, sort_keys=True, separators=(',', ':')).encode()),
              'split_plan_sha256': sha((OUT / 'split_plan.json').read_bytes()),
              'sources': list(sources.values()), 'notes': notes}
    write_json(OUT / 'annotation_audit.json', report)
    print(json.dumps({key: report[key] for key in ['source_count', 'extracted_txt_count',
                     'annotation_passed_sources', 'pending_source_ids', 'source_frames_with_consistent_labels',
                     'frame_annotations', 'actual_split_counts']}, ensure_ascii=False))
    for name in ['annotation_audit.json', 'split_plan.json']:
        path = OUT / name
        print(name, path.stat().st_size, sha(path.read_bytes()))


if __name__ == '__main__':
    main()
