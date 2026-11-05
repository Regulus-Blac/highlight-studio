"""Verify or replay a complete, frozen shared TimeLens release offline.

``cache-replay`` reconstructs every frame/box from fixed decoded rows, original
detection evidence and learned centers. It never loads a neural model or media.
``prepare`` hashes the original media, model and processor files for a future
deployment; it also performs no neural inference. All paths are CLI arguments.
Neither phase trains, changes the method, downloads, submits or resumes a job.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

from aic.evaluation.predictions import prediction_errors


SCHEMA = 'aic_shared_TimeLens_V8_portable_v1'
PARAMETERS = 8827283814


def relative(value):
    if not isinstance(value, str) or not value or value == '.':
        raise ValueError('A nonempty portable relative path is required')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value or '\0' in value:
        raise ValueError('A portable relative path is required: ' + value)
    return p


def located(root, name):
    p = (Path(root) / relative(name)).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Input leaves its declared root: ' + name)
    return p


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            check()
            h.update(block)
    return h.hexdigest()


def verified(root, entry, check):
    check()
    path = located(root, entry['path'])
    if path.stat().st_size != entry['bytes'] or sha(path, check) != entry['sha256']:
        raise ValueError('Frozen input changed: ' + entry['path'])
    return path


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_recipe(path):
    recipe = json.loads(Path(path).read_bytes())
    sources = recipe.get('sources', [])
    if (recipe.get('schema') != SCHEMA or recipe.get('whole_pipeline_parameters') != PARAMETERS
            or len(sources) != 426 or len({str(s['record']['video_id']) for s in sources}) != 426
            or len(recipe.get('detection_cache', [])) != 254
            or recipe.get('geometry_counts') != {'full': 172, 'x': 123, 'y': 131}):
        raise ValueError('Complete frozen V8 release recipe required')
    if len(recipe['source_sha256']) != 31 or not recipe.get('wrapper_sha256'):
        raise ValueError('All original sources and portable wrappers must be bound')
    return recipe


def source_code(args, recipe, check):
    for name, digest in recipe['source_sha256'].items():
        if sha(located(args.source_dir, name), check) != digest:
            raise ValueError('Original production source changed: ' + name)
    project = Path(__file__).resolve().parents[2]
    for name, digest in recipe['wrapper_sha256'].items():
        if sha(located(project, name), check) != digest:
            raise ValueError('Portable wrapper changed: ' + name)


@contextmanager
def original_modules(root):
    """Import only the locked upstream helpers, then restore module search paths."""
    before = list(sys.path)
    names = ('reframe', 'trim_frames', 'baseline_qwen', 'highlight', 'highlight.geometry')
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path[:0] = [str(Path(root) / 'tools'), str(root)]
    try:
        yield importlib.import_module('reframe'), importlib.import_module('trim_frames'), importlib.import_module('highlight.geometry')
    finally:
        sys.path[:] = before
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def read_rows(path, ids):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()]
    by_id = {str(row['video_id']): row for row in rows}
    if len(rows) != len(ids) or len(by_id) != len(ids) or set(by_id) != set(ids):
        raise ValueError('Missing, duplicate or unexpected input rows: ' + str(path))
    if any('error' in row for row in rows):
        raise ValueError('Failed row is not eligible: ' + str(path))
    return by_id


def full_entry(base, shifted, n, fps, trim_full, trim_row):
    """The original frame_probe gates on the already-trimmed baseline row."""
    empty_evidence = {'mean_diff': [], 'dets': [], 'frames': []}
    original, _ = trim_row(base, empty_evidence, n, fps, trim_full)
    if not original['predictions']:
        return original
    return trim_row(shifted, empty_evidence, n, fps, trim_full)[0]


def render_fixed_method(args, recipe, check):
    """Render a frozen method's complete inputs, without claiming an old SHA.

    A new, explicitly unscored engineering variant has no prior payload SHA.
    The strict V8 replay API below still enforces its original expected hash.
    """
    import numpy as np
    root = args.cache_root
    ids = {str(s['record']['video_id']) for s in recipe['sources']}
    inputs = {name: verified(root, entry, check) for name, entry in recipe['cache_inputs'].items()}
    decoded = [read_rows(inputs['shift' + str(i)], ids) for i in (0, 3)]
    source_by_id = {str(s['record']['video_id']): s for s in recipe['sources']}
    for rows in decoded:
        for vid, row in rows.items():
            errors = prediction_errors(row, source_by_id[vid]['record'], source_by_id[vid]['metadata'])
            if errors:
                raise ValueError('Invalid decoded input: ' + vid + ': ' + repr(errors[:3]))
    det_entries = {item['video_id']: item for item in recipe['detection_cache']}
    if len(det_entries) != 254:
        raise ValueError('Duplicate detection evidence identities')
    centers = {}
    with original_modules(args.source_dir) as (rf, tf, geometry):
        groups = {axis: set() for axis in ('full', 'x', 'y')}
        for vid, source in source_by_id.items():
            m = source['metadata']
            g = rf.geometry(m['width'], m['height'], source['record']['targetRatioWH'])
            groups['full' if g is None else g[0]].add(vid)
        if {k: len(v) for k, v in groups.items()} != recipe['geometry_counts']:
            raise ValueError('Original geometry partition changed')
        if set(det_entries) != groups['x'] | groups['y']:
            raise ValueError('Detection evidence coverage differs')
        for axis in ('x', 'y'):
            data = read_rows(inputs['centers_' + axis], groups[axis])
            centers[axis] = {}
            for vid, row in data.items():
                c = np.asarray(row['center_' + axis], dtype=float)
                if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 1):
                    raise ValueError('Invalid learned centers: ' + vid)
                centers[axis][vid] = np.array([np.median(c[max(0, i - 1):i + 2]) for i in range(len(c))])
        config = json.loads((Path(args.source_dir) / 'tools/configs/final_AM.json').read_bytes())
        trim = tf.load_trim(config['trim'])
        trim_full = tf.load_trim(dict(config['trim'], tail_cap=.05, head=False))
        if trim_full['tail_base'] != .05 or trim_full['min_keep_s'] != 3:
            raise ValueError('Frozen full trim differs')
        out = args.out_dir / 'submission.jsonl'
        frames = sampled = 0
        with out.open('x', encoding='utf-8', newline='\n') as stream:
            for source in recipe['sources']:
                check()
                record, m = source['record'], source['metadata']
                vid = str(record['video_id'])
                n, fps, W, H = m['frames'], m['fps'], m['width'], m['height']
                base = decoded[0][vid]
                geo = rf.geometry(W, H, record['targetRatioWH'])
                if geo is None:
                    row = full_entry(base, decoded[1][vid], n, fps, trim_full, tf.trim_row)
                    boxes = rf.place_boxes([p['frame'] for p in row['predictions']], W, H, record['targetRatioWH'])
                else:
                    path = verified(root, det_entries[vid], check)
                    cache = json.loads(path.read_bytes())
                    selected = [p['frame'] for p in base['predictions']]
                    sampled_frames = rf.sample_frames(selected, config['reframe']['stride'])
                    if (cache['frames'] != sampled_frames or cache['W'] != W or cache['H'] != H
                            or cache['detector'] != det_entries[vid]['detector']
                            or len(cache['dets']) != len(sampled_frames) or len(cache['mean_diff']) != len(sampled_frames)):
                        raise ValueError('Detection input sampling differs: ' + vid)
                    sampled += len(sampled_frames)
                    row, _ = tf.trim_row(base, cache, n, fps, trim)
                    c = centers[geo[0]][vid]
                    tw, th = map(float, record['targetRatioWH'])
                    w, h = geometry.safe_crop_size(W, H, tw, th)
                    travel = geometry.dof_range(W, H, tw, th)
                    length, extent = (W, w) if geo[0] == 'x' else (H, h)
                    boxes = {}
                    for p in row['predictions']:
                        cc = float(np.interp(p['frame'] / fps, np.arange(len(c)), c))
                        u = min(1., max(0., (cc * length - extent / 2.) / travel)) if travel > 0 else .5
                        boxes[p['frame']] = list(map(int, geometry.place_1d(u, W, H, tw, th)))
                preds = []
                for p in row['predictions']:
                    x, y, w = boxes[p['frame']]
                    if geo is None:
                        tw, th = record['targetRatioWH']
                        w2 = int(w * .95)
                        h, h2 = w * th / tw, w2 * th / tw
                        x = min(max(0, int(round(x + (w - w2) / 2))), W - w2)
                        y = min(max(0, int(round(y + (h - h2) / 2))), int(H - h2))
                        w = w2
                    preds.append(dict(p, bboxes=[int(x), int(y), int(w)]))
                row = dict(row, predictions=preds, model_params_b=PARAMETERS / 1e9)
                errors = prediction_errors(row, record, m, require_model_params=True)
                if errors:
                    raise ValueError('Replayed output invalid: ' + vid + ': ' + repr(errors[:3]))
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                frames += len(preds)
        actual = sha(out, check)
        return {'videos': len(ids), 'prediction_count': frames, 'sampled_detection_frames': sampled,
                'prediction_sha256': actual,
                'neural_reproduction_verified': False, 'input_media_or_labels_read': False,
                'original_decode_and_detection_caches_reused': True}


def replay(args, recipe, check):
    result = render_fixed_method(args, recipe, check)
    actual = result['prediction_sha256']
    if actual != recipe['expected_prediction_sha256']:
        raise ValueError('Complete replay differs from frozen V8 payload: ' + actual)
    out = args.out_dir / 'submission.jsonl'
    with zipfile.ZipFile(args.out_dir / 'results.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.write(out, arcname='submission.jsonl')
    with zipfile.ZipFile(args.out_dir / 'results.zip') as archive:
        if archive.namelist() != ['submission.jsonl'] or archive.testzip() is not None or hashlib.sha256(archive.read('submission.jsonl')).hexdigest() != actual:
            raise ValueError('Result container differs')
    return dict(result, complete_payload_matches_release=True,
                complete_payload_matches_V8=recipe.get('version', 'V8') == 'V8')


def prepare(args, recipe, check):
    roots = {'backbone': args.model, 'processor': args.processor, 'heads': args.heads}
    checked = []
    for group, root in roots.items():
        if root is None:
            raise ValueError('--model, --processor and --heads are required for prepare')
        for entry in recipe['assets'][group]:
            verified(root, entry, check)
            checked.append(group + '/' + entry['path'])
    detector = recipe['assets']['detector']
    if args.detector is None or args.detector.stat().st_size != detector['bytes'] or sha(args.detector, check) != detector['sha256']:
        raise ValueError('Original trim detector required')
    if args.index is None or args.video_dir is None:
        raise ValueError('--index and --video-dir are required for prepare')
    records = [json.loads(line) for line in args.index.read_text(encoding='utf-8-sig').splitlines()]
    if records != [source['record'] for source in recipe['sources']]:
        raise ValueError('Complete original index and order required')
    for source in recipe['sources']:
        entry = dict(path=str(source['record']['video_id']) + '.mp4', bytes=source['bytes'], sha256=source['sha256'])
        verified(args.video_dir, entry, check)
    versions = {}
    for name, expected in recipe['environment_packages'].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError('Frozen runtime dependency differs: ' + name)
        versions[name] = actual
    return {'verified_asset_files': len(checked) + 1, 'verified_media_files': len(records),
            'environment_packages': versions, 'model_instances': 0, 'neural_reproduction_verified': False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('cache-replay', 'prepare'), required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--processor', type=Path)
    p.add_argument('--heads', type=Path)
    p.add_argument('--detector', type=Path)
    p.add_argument('--index', type=Path)
    p.add_argument('--video-dir', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--deadline-unix', type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.max_seconds <= 21600:
        raise ValueError('Explicit wall limit must be in 1..21600 seconds')
    started = time.time()
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        raise ValueError('Finite deadline required')
    end = min(started + args.max_seconds,
              args.deadline_unix if args.deadline_unix is not None else float('inf'))

    def check():
        if time.time() >= end:
            raise TimeoutError('Original execution wall limit reached')

    if args.phase == 'cache-replay' and args.cache_root is None:
        raise ValueError('--cache-root is required for cache-replay')
    if args.out_dir.exists():
        raise FileExistsError('A new output directory is required')
    check()
    recipe = load_recipe(args.recipe)
    source_code(args, recipe, check)
    args.out_dir.mkdir(parents=False, exist_ok=False)
    state = {'active': True, 'pid': __import__('os').getpid(), 'phase': args.phase,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'maximum_seconds': args.max_seconds,
             'hard_deadline_unix': end, 'neural_calls': 0, 'training': False, 'submitted': False}
    write(args.out_dir / 'experiment.json', state)
    error = None
    try:
        result = replay(args, recipe, check) if args.phase == 'cache-replay' else prepare(args, recipe, check)
        check()
        write(args.out_dir / 'result.json', result)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        raise
    finally:
        state.update(active=False, complete=error is None, error=error,
                     elapsed_seconds=time.time() - started, ended_utc=datetime.now(timezone.utc).isoformat())
        (args.out_dir / 'experiment.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        write(args.out_dir / 'terminal.json', state)
    print(json.dumps(dict(state, result=result), ensure_ascii=False))


if __name__ == '__main__':
    main()
