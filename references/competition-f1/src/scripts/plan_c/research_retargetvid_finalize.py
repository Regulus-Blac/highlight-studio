#!/usr/bin/env python3
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('outputs/research_44_76/retargetvid_audit/full')
REFS = Path('outputs/research_1022/exclusion_hashes')
POP = np.array([i.bit_count() for i in range(256)], np.uint8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references(paths):
    hashes, owners, offsets, starts, exact = [], [], [], [], {}
    for path in paths:
        data = json.loads(path.read_text())
        starts.append(len(hashes)); hashes.extend(data['phash'])
        owners.extend([path.stem] * len(data['phash'])); offsets.extend(range(len(data['phash'])))
        exact[data['sha256']] = path.stem
    return np.array(hashes, np.uint8), owners, offsets, np.array(starts), exact


def match(signature, refs):
    hashes, owners, offsets, starts, exact = refs
    if signature['sha256'] in exact:
        return {'method': 'sha256', 'reference': exact[signature['sha256']]}
    sample = np.array(signature['phash'], np.uint8)
    distance = POP[np.bitwise_xor(sample[:, None], hashes[None])].sum(-1)
    hits = distance <= 10
    previous = np.zeros(len(hashes), np.int16)
    for i, row in enumerate(hits):
        one = np.zeros_like(previous); one[1:] = previous[:-1]; one[starts] = 0
        two = np.zeros_like(previous); two[2:] = previous[:-2]; two[starts] = 0
        two[(starts + 1)[starts + 1 < len(two)]] = 0
        run = row * (1 + np.maximum(one, two))
        found = np.flatnonzero(run >= 3)
        if len(found):
            j = int(found[0])
            return {'method': 'continuous_phash_3_seconds', 'reference': owners[j],
                    'end_sample_seconds': [i, offsets[j]], 'maximum_hamming': 10,
                    'reference_step_per_second': [1, 2]}
        previous = run
    return None


def labels(vid, media, folder):
    folder.mkdir(parents=True, exist_ok=True)
    raw = {}
    for ratio in ('1-3', '3-1'):
        raw[ratio] = np.stack([np.loadtxt(ROOT / 'annotations' / f'annotator_{a}' / f'{vid}_{ratio}.txt',
                                         delimiter=',', dtype=np.int32) for a in range(1, 7)], axis=1)
    n, width, height = media['n_frames'], media['width'], media['height']
    assert raw['1-3'].shape == raw['3-1'].shape == (n, 6, 4)
    crop_w = int(height * 9 / 16)
    crop_h = crop_w * 16 / 9
    raw_center = (raw['1-3'][..., 0] + raw['1-3'][..., 2]) / 2
    x = np.clip(np.rint(raw_center - crop_w / 2), 0, width - crop_w).astype(np.int32)
    y = min(max(0, round(height / 2 - crop_h / 2)), int(height - crop_h))
    derived = np.stack([x, np.full_like(x, y), np.full_like(x, crop_w)], axis=-1)
    assert np.all(x >= 0) and np.all(x + crop_w <= width) and y + crop_h <= height
    path = folder / (vid + '.npz')
    np.savez_compressed(path, centers=((x + crop_w / 2) / width).astype(np.float32),
                        raw_centers=(raw_center / width).astype(np.float32),
                        intervals=np.stack([x / width, (x + crop_w) / width], -1).astype(np.float32),
                        valid=np.ones((n, 6), bool), frame_indices=np.arange(n, dtype=np.int64),
                        times=np.arange(n) / media['fps'], annotator_ids=np.arange(1, 7, dtype=np.uint8),
                        original_1x3_xyxy=raw['1-3'], original_3x1_xyxy=raw['3-1'], derived_9x16_xyw=derived)
    return path


def main():
    audit = json.loads((ROOT / 'annotation_audit.json').read_text())
    plan = json.loads((ROOT / 'split_plan.json').read_text())
    source_ids = sorted(plan['source_to_split'])
    planned_counts = {s: len(plan['splits'][s]) for s in ('train', 'dev', 'heldout')}
    assert len(source_ids) == 200 and planned_counts == {'train': 160, 'dev': 20, 'heldout': 20}
    media = {v: json.loads((ROOT / 'metadata' / (v + '.json')).read_text()) for v in source_ids}
    assert all(m['status'] == 'complete' and m['crc32_match'] for m in media.values())
    annotations = {s['source_id']: s for s in audit['sources']}
    assert all(annotations[v]['annotation_status'] == 'passed' and annotations[v]['n_frames'] == media[v]['n_frames']
               for v in source_ids)
    paths = sorted(REFS.glob('*.json'))
    assert len(paths) == 3267
    refs = references(paths)
    excluded, rows, errors, pool_paths = [], [], [], []
    priority = {'heldout': 0, 'dev': 1, 'train': 2}
    ordered_ids = sorted(source_ids, key=lambda v: (priority[plan['source_to_split'][v]], v))
    for n, vid in enumerate(ordered_ids, 1):
        m = media[vid]
        try:
            assert sha(Path(m['video_path'])) == m['sha256']
            found = match(m, refs)
            scope = 'existing3267_reference'
            if not found and pool_paths:
                found = match(m, references(pool_paths))
                scope = 'official200_source_pool'
            if found:
                excluded.append({'source_id': vid, 'dataset_split': plan['source_to_split'][vid], 'scope': scope, **found})
                print('RETARGET_EXCLUDED', vid, found['method'], found['reference'], flush=True)
                continue
            target = labels(vid, m, ROOT / 'labels')
            indices = [round(k * m['fps']) for k in range(int((m['n_frames'] - 1) // m['fps']) + 1)]
            rows.append({'video_id': 'retarget_dhf_' + vid, 'source_group': 'DHF1K_' + vid,
                         'dataset_split': plan['source_to_split'][vid], 'video_path': m['video_path'],
                         **{k: m[k] for k in ('width', 'height', 'fps', 'n_frames', 'duration')},
                         'targetRatioWH': [9, 16], 'sampling_frame_indices': indices,
                         'references_path': str(target), 'references_sha256': sha(target), 'annotators': [1, 2, 3, 4, 5, 6],
                         'annotation_type': 'human_derived_9x16_from_1x3', 'data_ready': True,
                         'provenance': {'source_group': 'DHF1K_' + vid, 'source_identity': 'official_DHF1K_RAR_member',
                                        'official_member': m['member'], 'video_sha256': m['sha256'],
                                        'official_member_crc32': m['crc32'], 'media_license': 'CC-BY-4.0',
                                        'annotations_license': 'RetargetVid repository MIT; no separate annotation-specific grant found; applicability to annotations inferred',
                                        'native_annotation_ratios': ['1:3', '3:1'],
                                        'media_source': 'https://github.com/wenguanwang/DHF1K',
                                        'annotation_source': 'https://github.com/bmezaris/RetargetVid'},
                         'dedup': {'method': 'SHA256 or3consecutive1fps phash; Hamming<=10; reference steps1or2',
                                   'reference_videos': 3267, 'reference_frame_samples': len(refs[0]),
                                   'within_pool_policy': 'heldout before dev before train, then ascending officialID; no refill'}})
            pool_paths.append(ROOT / 'metadata' / (vid + '.json'))
            print('RETARGET_QUALIFIED', n, len(source_ids), vid, m['n_frames'], flush=True)
        except Exception as e:
            error = {'source_id': vid, 'predictions': [], 'error': str(e)}
            errors.append(error); print('error', json.dumps(error), flush=True)
    rows.sort(key=lambda r: r['video_id'])
    inputs = {'script_sha256': sha(Path(__file__)), 'annotation_audit_sha256': sha(ROOT / 'annotation_audit.json'),
              'split_plan_sha256': sha(ROOT / 'split_plan.json'), 'official_index_sha256': sha(ROOT / 'official_archive_index.jsonl'),
              'recovery_report_sha256': sha(ROOT / 'recovery_report.json'),
              'official_source_evidence_sha256': {p.name: sha(p) for p in
                  [ROOT.parent / 'source' / s for s in ('official_sources.json', 'RetargetVid_README.md',
                   'RetargetVid_LICENSE.md', 'DHF1K_README.md', 'generate_frame.m')]},
              'reference_metadata_sha256': {str(p): sha(p) for p in paths},
              'media_metadata_sha256': {v: sha(ROOT / 'metadata' / (v + '.json')) for v in source_ids}}
    (ROOT / 'qualification_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    for split in ('train', 'dev', 'heldout', 'all_clean'):
        subset = rows if split == 'all_clean' else [r for r in rows if r['dataset_split'] == split]
        (ROOT / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in subset))
    (ROOT / 'dedup_excluded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in excluded))
    (ROOT / 'qualification_errors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in errors))
    report = {'status': 'READY' if not errors else 'ERROR', 'ready': not errors,
              'source_count': 200, 'planned_split_counts': planned_counts,
              'qualified_source_count': len(rows), 'split_counts': {s: sum(r['dataset_split'] == s for r in rows) for s in planned_counts},
              'canonical_sources_checked': 200, 'source_frames_before_exclusion': sum(m['n_frames'] for m in media.values()),
              'qualified_sources': len(rows), 'actual_split_counts': dict(Counter(r['dataset_split'] for r in rows)),
              'actual_split_frames': {s: sum(r['n_frames'] for r in rows if r['dataset_split'] == s) for s in ('train', 'dev', 'heldout')},
              'excluded_sources': len(excluded), 'excluded_ids': [r['source_id'] for r in excluded], 'errors': errors,
              'reference_videos': len(paths), 'reference_frame_hashes': len(refs[0]), 'test_media_decoded': 0,
              'media_license': 'CC-BY-4.0 (DHF1K author README)',
              'annotations_license': 'RetargetVid repository MIT; no separate annotation-specific grant found; applicability to annotations inferred',
              'qualification': {'official_media_identity': 'exact author-linked RAR member name/header/payload',
                                'crc32': 'all200 matched original member CRC32',
                                'frame_count': 'all200 decoded to EOF and matched all6 annotators at both native ratios',
                                'frame_mapping': 'annotation row0 maps canonical decoded frame0, t=frame/fps; no FPS resampling',
                                'content_exclusion': 'SHA256 and3 consecutive1FPS pHash Hamming<=10, reference step1or2, against3267cached references and accepted official200sources; not a full provenance-independence proof'},
              'training_started': False, 'gpu_used': False, 'qualification_inputs_sha256': sha(ROOT / 'qualification_inputs.json'),
              'all_clean_sha256': sha(ROOT / 'all_clean.jsonl'),
              'labels_sha256': {r['video_id']: r['references_sha256'] for r in rows},
              'files_sha256': {str(p): sha(p) for p in list((ROOT / 'labels').glob('*.npz')) +
                              [ROOT / (s + '.jsonl') for s in ('train', 'dev', 'heldout', 'all_clean')]},
              'notes': ['Source split frozen160/20/20 before media recovery and pHash exclusion; never refill excluded sources.',
                        'Within-pool exclusion priority fixed before matching: heldout, dev, train, then ascending officialID.',
                        'Six annotators retained separately per original frame and original DHF1K source group; no averaging.',
                        '9:16 is derived from original1:3; native1:3/3:1integer xyxy retained unchanged.',
                        'Only cached AIC/reference SHA/pHash metadata read; no test/blind media decoded or sent to API.',
                        'Content exclusion against3267references is not full original-author/source-identity independence proof.',
                        'Heldout labels were read for geometry/length qualification only, without model outputs or scoring.']}
    (ROOT / 'qualification_full_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['ready']:
        (ROOT / 'READY.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OFFICIAL_RETARGET_QUALIFICATION_COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}), flush=True)
    assert report['ready']


if __name__ == '__main__':
    main()
