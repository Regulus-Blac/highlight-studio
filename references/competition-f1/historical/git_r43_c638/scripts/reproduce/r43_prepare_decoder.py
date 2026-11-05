"""Record capability metadata before the original R43 CPU decoder qualification.

This packages the original preparation step. It does not load model weights,
view test images, generate labels, select video-specific rules or train.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from r43_readers import OUT, reliable


def main():
    test = Path('/root/sj-fs/autodl-tmp/public/aic-competition-test/2026_semifinal')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=test/'submission_index.jsonl')
    parser.add_argument('--video-dir', type=Path, default=test/'video')
    parser.add_argument('--out', type=Path, default=OUT/'decoder_metadata.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ids = [str(row['video_id']) for row in rows]
    if len(ids) != 426 or len(set(ids)) != 426:
        raise ValueError('The complete426-source index without duplicate identities is required')
    if args.out.exists():
        raise FileExistsError('Preserve the previous metadata: ' + str(args.out))
    video_dir = args.video_dir.resolve()
    metadata = {}
    for vid in ids:
        path = (video_dir/(vid+'.mp4')).resolve()
        if not path.is_relative_to(video_dir) or not path.is_file():
            raise ValueError('Missing or out-of-root video source')
        metadata[vid] = reliable.decoder_metadata(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('METADATA_COMPLETE', len(metadata), dict(Counter(m['backend'] for m in metadata.values())))


if __name__ == '__main__':
    main()
