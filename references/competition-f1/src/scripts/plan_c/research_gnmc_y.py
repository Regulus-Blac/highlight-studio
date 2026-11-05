#!/usr/bin/env python3
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image


ARCHIVE = Path('outputs/recovery/gnmc_crop_v1.zip')
OUT = Path('outputs/research_44_76/gnmc_y')
EXPECTED = {'train': 1461, 'validation': 209}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'source').mkdir()
archive_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
assert archive_sha == 'a8fe3417beedeea9194b8b117356a51829ec0f22f30547b61c166be61a48ac77'
prepared, hashes, counts = [], {}, {}
with zipfile.ZipFile(ARCHIVE) as z:
    for name in ('LICENSE.txt', 'manifest.json', 'README.original.md'):
        (OUT / 'source' / name).write_bytes(z.read(name))
    source = json.loads(z.read('manifest.json'))
    for split in ('train', 'validation'):
        original = list(map(json.loads, z.read(split + '.jsonl').decode().splitlines()))
        selected = [r for r in original if r['width'] / r['height'] < 16 / 9 - .02]
        assert len(selected) == EXPECTED[split]
        (OUT / 'images' / split).mkdir(parents=True)
        rows, file_hashes, pixel_hashes, rgb_hashes = [], set(), set(), set()
        outside, ratio_error = 0, []
        for r in selected:
            blob = z.read(r['image_path'])
            image_sha = hashlib.sha256(blob).hexdigest()
            assert image_sha == r['image_sha256']
            image = Image.open(io.BytesIO(blob)).convert('RGB')
            assert image.size == (r['width'], r['height'])
            rgb_sha = hashlib.sha256(('%dx%d:' % image.size).encode() + image.tobytes()).hexdigest()
            path = OUT / 'images' / split / r['source_filename']
            path.write_bytes(blob)
            x1, y1, x2, y2 = r['original_crop_xyxy']
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
            error = abs((x2 - x1) * r['width'] / ((y2 - y1) * r['height']) / (16 / 9) - 1)
            assert error < .01
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            half = r['width'] / r['height'] * 9 / 32
            projected_y = min(max(center[1], half), 1 - half)
            assert half <= projected_y <= 1 - half
            outside += int(abs(projected_y - center[1]) > 1e-12)
            video_id = 'gnmc_' + split + '_' + Path(r['source_filename']).stem
            row = {'video_id': video_id, 'video_path': str(path), 'dataset_split': split,
                   'task': 'spatial_only', 'width': r['width'], 'height': r['height'],
                   'fps': 1, 'n_frames': 1, 'duration': 1, 'targetRatioWH': [16, 9],
                   'sampling_frame_indices': [0], 'portrait': r['height'] > r['width'],
                   'original_crop_xyxy': r['original_crop_xyxy'], 'crop_coordinate_system': 'normalized_xyxy',
                   'teacher_signals': {'trajectory': [{'time_sec': 0, 'center': center}]},
                   'max_size_center_y': projected_y, 'image_sha256': image_sha,
                   'pixel_sha256': r['pixel_sha256'], 'rgb_pixel_sha256': rgb_sha,
                   'label_source': r['label_source'], 'source': source['source'],
                   'provenance': {'source_group': 'gnmc_' + r['pixel_sha256'],
                                  'source_identity': 'pixel_hash_only', 'original_split': split,
                                  'original_sample_id': r['sample_id'], 'source_filename': r['source_filename'],
                                  'author_repository': source['author_repository'],
                                  'archive_sha256': archive_sha, 'image_sha256': image_sha,
                                  'pixel_sha256': r['pixel_sha256']}}
            rows.append(row)
            file_hashes.add(image_sha); pixel_hashes.add(r['pixel_sha256']); rgb_hashes.add(rgb_sha)
            ratio_error.append(error)
        (OUT / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        prepared.extend(rows)
        hashes[split] = (file_hashes, pixel_hashes, rgb_hashes)
        counts[split] = {'images': len(rows), 'portrait': sum(r['portrait'] for r in rows),
                         'raw_center_outside_max_size_range': outside, 'max_original_aspect_error': max(ratio_error),
                         'distinct_file_hashes': len(file_hashes), 'distinct_original_pixel_hashes': len(pixel_hashes),
                         'distinct_rgb_pixel_hashes': len(rgb_hashes)}
shared = {name: len(hashes['train'][i] & hashes['validation'][i])
          for i, name in enumerate(('file', 'original_pixel', 'rgb_pixel'))}
assert not any(shared.values())
(OUT / 'all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prepared))
report = {'archive': str(ARCHIVE), 'archive_sha256': archive_sha, 'counts': counts,
          'train_validation_shared_hashes': shared, 'test_images_read': 0,
          'selection': 'source width / height < 16/9 - 0.02; original official train/validation kept',
          'source': source['source'], 'license': source['license'],
          'notes': ['已查看几何统计的公开验证，不是blind。',
                    'teacher center保留原人工框中点；max_size_center_y另存最大宽合法中心投影，不替换原GT。',
                    'pixel_sha256保留原值；rgb_pixel_sha256为本轮widthxheight前缀加RGB像素的独立核查。',
                    '精确像素隔离不是完整来源身份隔离；AIC训练/测试去重尚待完成。',
                    '静态图片只有空间标签，不构造高光真值；video_path为repo相对JPEG路径。']}
(OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'files_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
package = OUT.with_suffix('.tar')
with tarfile.open(package, 'x') as tar:
    tar.add(OUT, arcname=str(OUT))
receipt = {'tar': str(package), 'bytes': package.stat().st_size,
           'sha256': hashlib.sha256(package.read_bytes()).hexdigest(), 'files': len(manifest) + 1,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
package.with_suffix('.tar.sha256.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(report, package=receipt), ensure_ascii=False, indent=2))
