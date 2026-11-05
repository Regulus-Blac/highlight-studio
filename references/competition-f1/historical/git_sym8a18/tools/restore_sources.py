#!/usr/bin/env python3
"""Restore the exact QVH sources from the pinned mirror; verify the old disk's SHA256 manifest."""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    manifest = json.load(open('outputs/recovery/qvh_source_manifest.json'))
    candidates = json.load(open('outputs/recovery/recovery_candidates.json'))
    root = Path('/root/sj-fs/autodl-tmp/public/qvhighlights/videos')
    base = ('https://hf-mirror.com/datasets/ayushsdev/qvhighlights-videos/resolve/'
            + manifest['video_revision'] + '/')

    def fetch(row):
        rel = row['path'].split('/videos/')[1]
        dst = root / rel
        if dst.exists() and sha(dst) == row['sha256']:
            return 'verified'
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix('.http.part')
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--connect-timeout', '20',
                        '--max-time', '600', '-o', str(part), base + rel], check=True)
        assert sha(part) == row['sha256'], rel + ': SHA256 mismatch'
        part.replace(dst)
        return 'downloaded'

    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, row): row['path'] for row in candidates['training_sources']}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                status = future.result()
            except Exception as e:
                errors.append(futures[future])
                status = 'error: ' + str(e)
            print('[%d/987] %s %s' % (i, Path(futures[future]).name, status), flush=True)
    assert not errors, errors
    print('SOURCES_VERIFIED_987', flush=True)


if __name__ == '__main__':
    main()
