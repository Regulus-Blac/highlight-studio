"""Run a pinned original shared extractor with capability-checked video input."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from aic.data.reliable_video import spatial_reader, temporal_reader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('temporal', 'spatial'), required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ['--']:
        forwarded.pop(0)
    source = args.source_dir.resolve()
    recipe = json.loads(args.recipe.read_bytes())
    for name, expected in recipe['source_sha256'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original shared source differs: ' + name)
    for directory in (source, source / 'baseline', source / 'tools'):
        sys.path.insert(0, str(directory))
    if args.kind == 'spatial':
        import api_teacher
        api_teacher.frames_1fps = spatial_reader(api_teacher.frames_1fps)
    script = source / 'tools' / ('infer_yesno.py' if args.kind == 'temporal' else 'spatial_feats.py')
    spec = importlib.util.spec_from_file_location('_aic_checked_shared_' + args.kind, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.kind == 'temporal':
        module.read_frames = temporal_reader(module.read_frames)
    sys.argv = [str(script), *forwarded]
    result = module.main()
    if result:
        raise SystemExit(result)


if __name__ == '__main__':
    main()
