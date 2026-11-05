#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time
import zlib

import cv2
import numpy as np
import requests


ROOT = Path('outputs/research_44_76/retargetvid_audit')
OUT = ROOT / 'full'
URL = 'https://drive.usercontent.google.com/download?id=1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s&export=download&confirm=t'
TOTAL = 4032563381
TARGETS = {f'{i:03d}' for i in list(range(1, 101)) + list(range(601, 701))}
LOCAL = threading.local()
PRINT_LOCK = threading.Lock()


def fetch(start, size):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    for attempt in range(3):
        try:
            response = LOCAL.session.get(URL + f'&range_start={start}&range_size={size}',
                headers={'Range': f'bytes={start}-{start + size - 1}'}, timeout=(10, 45))
            response.raise_for_status()
            assert response.status_code == 206, 'Expected public Range response; do not log in or accept terms'
            assert response.headers['Content-Range'] == f'bytes {start}-{start + size - 1}/{TOTAL}'
            assert len(response.content) == size
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def recover(item, main_header):
    vid = item['source_id']
    meta_path = OUT / 'metadata' / (vid + '.json')
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous['status'] == 'complete':
            assert hashlib.sha256(Path(previous['video_path']).read_bytes()).hexdigest() == previous['sha256']
            return previous
    try:
        assert not item['flags'] & (1 | 2 | 4 | 16 | 256), 'split, encrypted, solid or >4GB member needs full archive'
        path = OUT / 'videos' / (vid + '.AVI')
        sample = ROOT / 'official_samples/video' / (vid + '.AVI')
        if not path.exists() and sample.exists():
            shutil.copyfile(sample, path)
        if not path.exists():
            packed = OUT / 'packed' / (vid + '.rar')
            if not packed.exists():
                payload = fetch(item['offset'], item['header_size'] + item['packed_bytes'])
                assert hashlib.sha256(payload[:item['header_size']]).hexdigest() == item['header_sha256']
                end = b'\x7b' + struct.pack('<HH', 0x4000, 7)
                packed.write_bytes(main_header + payload + struct.pack('<H', zlib.crc32(end) & 0xffff) + end)
            with path.with_suffix('.AVI.part').open('wb') as f:
                subprocess.run(['/usr/bin/tar', '-xOf', str(packed), item['member']], stdout=f, check=True)
            path.with_suffix('.AVI.part').rename(path)
        blob = path.read_bytes()
        assert len(blob) == item['unpacked_bytes'] and zlib.crc32(blob) & 0xffffffff == item['crc32']
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame, next_sample, phash = 0, 0., []
        while cap.grab():
            if frame >= next_sample:
                ok, image = cap.retrieve()
                assert ok
                gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
                dct = cv2.dct(gray)[:8, :8].flatten(); bits = dct > np.median(dct[1:]); bits[0] = False
                phash.append(np.packbits(bits).tolist()); next_sample += fps
            frame += 1
        cap.release()
        assert frame == reported and width == 640 and height == 360
        result = dict(item, status='complete', video_path=str(path), sha256=hashlib.sha256(blob).hexdigest(),
                      crc32_match=True, width=width, height=height, fps=fps, n_frames=frame,
                      duration=frame / fps, phash=phash, phash_method='32x32-DCT-8x8-DC-zero-1fps')
    except Exception as e:
        result = dict(item, status='error', error=str(e), predictions=[])
    meta_path.write_text(json.dumps(result, indent=2) + '\n')
    with PRINT_LOCK:
        print('RETARGET_SOURCE', vid, result['status'], result.get('n_frames'), result.get('error', ''), flush=True)
    return result


def main():
    cv2.setNumThreads(1)
    for name in ('metadata', 'videos', 'packed'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    prefix = (ROOT / 'official_samples/video.rar.bytes_0_5996536').read_bytes()[:20]
    assert prefix[:7] == b'Rar!\x1a\x07\x00'
    index_path = OUT / 'official_archive_index.jsonl'
    entries = list(map(json.loads, index_path.read_text().splitlines())) if index_path.exists() else []
    position = entries[-1]['next_offset'] if entries else 20
    found = {e['source_id']: e for e in entries if e.get('source_id') in TARGETS}
    futures = []
    with ThreadPoolExecutor(4) as pool, index_path.open('a') as log:
        for item in found.values():
            futures.append(pool.submit(recover, item, prefix))
        while set(found) != TARGETS:
            block = fetch(position, 1024)
            crc, kind, flags, size = struct.unpack_from('<HBHH', block)
            header = block[:size]
            assert zlib.crc32(header[2:]) & 0xffff == crc
            data_size = struct.unpack_from('<I', block, 7)[0] if flags & 0x8000 else 0
            entry = {'offset': position, 'header_type': kind, 'header_size': size, 'flags': flags,
                     'next_offset': position + size + data_size}
            if kind == 0x74:
                packed, unpacked, host, file_crc, stamp, version, method, name_size, attr = struct.unpack_from('<IIBIIBBHI', block, 7)
                member = block[32:32 + name_size].decode().replace('\\', '/')
                match = re.fullmatch(r'video/(\d+)\.AVI', member)
                entry.update(member=member, packed_bytes=packed, unpacked_bytes=unpacked, crc32=file_crc,
                             compression_method=method, header_sha256=hashlib.sha256(header).hexdigest())
                if match:
                    entry['source_id'] = f'{int(match[1]):03d}'
                    if entry['source_id'] in TARGETS:
                        found[entry['source_id']] = entry
                        futures.append(pool.submit(recover, entry, prefix))
            log.write(json.dumps(entry) + '\n'); log.flush(); entries.append(entry)
            position = entry['next_offset']
            if len(entries) % 25 == 0:
                print('OFFICIAL_RAR_INDEX', len(entries), 'target_headers', len(found), 'offset', position, flush=True)
            if kind == 0x7b or position >= TOTAL:
                break
        results = [future.result() for future in as_completed(futures)]
    errors = [r for r in results if r['status'] != 'complete']
    report = {'official_archive_file_id': '1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s', 'official_archive_bytes': TOTAL,
              'method': 'official RAR member headers plus individual non-solid member Range; no proxy/media resampling',
              'required_sources': 200, 'source_headers_found': len(found), 'completed_sources': len(results) - len(errors),
              'missing_ids': sorted(TARGETS - set(found)), 'error_ids': [r['source_id'] for r in errors],
              'video_bytes': sum(Path(r['video_path']).stat().st_size for r in results if r['status'] == 'complete'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_used': False, 'training_started': False, 'data_pending': True}
    (OUT / 'recovery_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'recovery_errors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in errors))
    print('OFFICIAL_RETARGET_RECOVERY_COMPLETE', json.dumps(report), flush=True)
    assert report['completed_sources'] == 200 and not report['missing_ids'] and not errors


if __name__ == '__main__':
    main()
