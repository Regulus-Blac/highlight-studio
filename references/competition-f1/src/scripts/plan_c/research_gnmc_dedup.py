#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path('outputs/research_44_76/gnmc_y')
REFERENCES = Path('outputs/research_1022/exclusion_hashes')
THRESHOLD = 8
POP = np.array([n.bit_count() for n in range(256)], np.uint8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phash(path):
    image = cv2.imread(str(path))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8].flatten()
    bits = dct > np.median(dct[1:]); bits[0] = False
    return np.packbits(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cv2.setNumThreads(1)
    out = ROOT / ('dedup_smoke' if args.smoke else 'dedup')
    out.mkdir(exist_ok=False)
    rows = list(map(json.loads, (ROOT / 'all.jsonl').open()))
    if args.smoke:
        rows = [next(r for r in rows if r['dataset_split'] == split) for split in ('train', 'validation')]
    inputs = {str(p): sha(p) for p in sorted(REFERENCES.glob('*.json'))}
    assert len(inputs) == 3267
    refs, ref_ids, ref_frames, ref_sha = [], [], [], {}
    for name in inputs:
        r = json.loads(Path(name).read_text())
        ref_sha[r['sha256']] = Path(name).stem
        refs.extend(r['phash'])
        ref_ids.extend([Path(name).stem] * len(r['phash']))
        ref_frames.extend(range(len(r['phash'])))
    refs = np.array(refs, np.uint8)
    assert refs.shape[1] == 8
    signatures, excluded = {}, {}
    for n, row in enumerate(rows, 1):
        v, path = row['video_id'], Path(row['video_path'])
        image_sha = sha(path)
        assert image_sha == row['image_sha256']
        signature = phash(path)
        signatures[v] = signature
        reasons = []
        if image_sha in ref_sha:
            reasons.append({'method': 'reference_sha256', 'reference': ref_sha[image_sha]})
        distance = POP[np.bitwise_xor(refs, signature)].sum(-1)
        k = int(distance.argmin())
        if int(distance[k]) <= THRESHOLD:
            reasons.append({'method': 'reference_single_image_phash', 'reference': ref_ids[k],
                            'reference_sample_index': ref_frames[k], 'hamming': int(distance[k])})
        if reasons:
            excluded[v] = reasons
        if n % 100 == 0 or n == len(rows):
            print('GNMC_REFERENCE_CHECK', n, len(rows), 'excluded', len(excluded), flush=True)
    validation = [r for r in rows if r['dataset_split'] == 'validation']
    val_hashes = np.array([signatures[r['video_id']] for r in validation], np.uint8)
    for row in rows:
        if row['dataset_split'] != 'train':
            continue
        distance = POP[np.bitwise_xor(val_hashes, signatures[row['video_id']])].sum(-1)
        k = int(distance.argmin())
        if int(distance[k]) <= THRESHOLD:
            excluded.setdefault(row['video_id'], []).append({'method': 'gnmc_train_validation_phash',
                'reference': validation[k]['video_id'], 'hamming': int(distance[k])})
    kept = []
    for row in rows:
        if row['video_id'] not in excluded:
            row['dedup'] = {'method': 'sha256+single_image_phash_hamming_le8',
                            'reference_count': len(inputs), 'gnmc_validation_count': len(validation)}
            kept.append(row)
    for split in ('train', 'validation', 'all'):
        subset = kept if split == 'all' else [r for r in kept if r['dataset_split'] == split]
        (out / (split + '_clean.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in subset))
    (out / 'excluded.jsonl').write_text(''.join(json.dumps({'video_id': v, 'reasons': reasons}) + '\n'
                                             for v, reasons in excluded.items()))
    input_hashes = {'script_sha256': sha(Path(__file__)), 'rows_sha256': sha(ROOT / 'all.jsonl'),
                    'reference_metadata_sha256': inputs,
                    'gnmc_images_sha256': {r['video_path']: r['image_sha256'] for r in rows}}
    (out / 'input_hashes.json').write_text(json.dumps(input_hashes, indent=2) + '\n')
    report = {'smoke_only': args.smoke, 'threshold': THRESHOLD, 'reference_videos': len(inputs),
              'reference_hash_samples': len(refs), 'input': dict(Counter(r['dataset_split'] for r in rows)),
              'kept': dict(Counter(r['dataset_split'] for r in kept)), 'excluded_images': len(excluded),
              'excluded_reasons': dict(Counter(reason['method'] for reasons in excluded.values() for reason in reasons)),
              'portrait_kept': dict(Counter(r['dataset_split'] for r in kept if r['portrait'])),
              'test_media_decoded': 0, 'input_hashes_sha256': sha(out / 'input_hashes.json'),
              'notes': ['只解码GNMC公开图片；比赛参考只读已有SHA/pHash JSON，不读取或查看测试视频。',
                        '统一32x32灰度DCT左上8x8，中位数阈值且DC置零，沿用research_dedup.py算法。',
                        '静态图单图Hamming<=8直接保守排除，可能误删；不采用三秒连续命中。',
                        'GNMC train/validation近重复仅从train排除；与外部参考命中时两划分均排除。',
                        '零命中不能证明完整来源身份独立；验证已查看几何，不是blind。']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    outputs = {p.name: sha(p) for p in sorted(out.glob('*')) if p.is_file()}
    (out / 'outputs_sha256.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print('GNMC_DEDUP_COMPLETE', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
