#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path('outputs/research_44_76/retargetvid_audit')
samples = json.loads((ROOT / 'official_samples/sample_alignment.json').read_text())
rows = []
for vid, meta in samples.items():
    video = Path(meta['path'])
    assert hashlib.sha256(video.read_bytes()).hexdigest() == meta['sha256']
    width, height, fps = meta['width'], meta['height'], meta['fps']
    crop_w = int(height * 9 / 16)
    crop_h = crop_w * 16 / 9
    for annotator in range(1, 7):
        folder = ROOT / 'official_samples/annotations' / f'annotator_{annotator}'
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / 'source' / f'annotator_{annotator}.zip') as archive:
            for ratio in ('1-3', '3-1'):
                name = f'{vid}_{ratio}.txt'
                raw = archive.read(f'annotator_{annotator}/' + name)
                (folder / name).write_bytes(raw)
        raw_path = folder / f'{vid}_1-3.txt'
        boxes = np.loadtxt(raw_path, delimiter=',', dtype=int)
        assert len(boxes) == meta['decoded_frames']
        annotations, trajectory = [], []
        for frame, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2
            x = min(max(0, round(cx - crop_w / 2)), width - crop_w)
            y = min(max(0, round(height / 2 - crop_h / 2)), int(height - crop_h))
            assert 0 <= x and x + crop_w <= width and 0 <= y and y + crop_h <= height
            annotations.append({'frame': frame, 'original_crop_xyxy': [x1, y1, x2, y2],
                                'derived_crop_xyw': [x, y, crop_w]})
            trajectory.append({'time_sec': frame / fps,
                               'center': [(x + crop_w / 2) / width, (y + crop_h / 2) / height]})
        rows.append({'video_id': f'retarget_dhf_{vid}_a{annotator}', 'source_video_id': f'DHF1K_{vid}',
                     'video_path': str(video), 'dataset_split': 'unassigned', 'official_DHF1K_split': 'train',
                     'width': width, 'height': height, 'fps': fps, 'n_frames': meta['decoded_frames'],
                     'duration': meta['decoded_frames'] / fps, 'targetRatioWH': [9, 16],
                     'sampling_frame_indices': list(range(meta['decoded_frames'])),
                     'annotation_type': 'human_derived_9x16_from_1x3', 'annotator': annotator,
                     'spatial_annotations': annotations, 'teacher_signals': {'trajectory': trajectory},
                     'provenance': {'source_group': f'DHF1K_{vid}', 'source_identity': 'official_DHF1K_archive_member',
                                    'official_member': f'video/{vid}.AVI', 'video_sha256': meta['sha256'],
                                    'original_annotation': str(raw_path),
                                    'annotation_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                    'license_media': 'CC-BY-4.0', 'annotation_repository_license': 'MIT',
                                    'media_source': 'https://github.com/wenguanwang/DHF1K',
                                    'annotation_source': 'https://github.com/bmezaris/RetargetVid'},
                     'dedup_pending': True, 'training_ready': False})
dest = ROOT / 'official_samples/human_derived_9x16_rows.jsonl'
dest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
manifest = {'source_videos': len(samples), 'annotators_per_video': 6, 'records': len(rows),
            'source_frames': sum(s['decoded_frames'] for s in samples.values()),
            'derived_frame_annotations': sum(r['n_frames'] for r in rows),
            'rows_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'split_policy': 'unassigned; all annotators/ratios of each original DHF1K ID must remain in one group',
            'conversion': 'center-preserving x expansion to maximum legal integer-width 9:16; clamp to frame; no resampling',
            'notes': ['9:16是人工1:3窗口的派生标签，不是原生9:16人工真值。',
                      '原六标注者的1:3与3:1原TXT均保留；不平均六人的轨迹。',
                      '本轮仅资格核验两公开源；尚未做AIC排重，不开始训练，也没有独立验证划分。']}
(dest.parent / 'samples_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('OFFICIAL_RETARGETVID_TWO_SOURCE_ASSET_READY', json.dumps(manifest, ensure_ascii=False))
