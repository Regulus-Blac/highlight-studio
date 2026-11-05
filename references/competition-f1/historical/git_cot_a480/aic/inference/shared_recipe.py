"""Plan or reproduce a frozen shared TimeLens pipeline offline.

Planning creates an inspectable complete command graph and dispatches nothing.
Running is a new, explicitly bounded reproduction in a new directory. It hashes
all original assets/media before any neural call, executes the original shared
sources serially, and accepts only the complete release's frozen prediction SHA.
The release recipe also binds its original or capability-based decoder policy.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from aic.deployment.shared_reproduction import (
    PARAMETERS, load_recipe, original_modules, prepare, read_rows, replay, sha,
    source_code, write,
)


def partitions(args, recipe):
    groups = {k: [] for k in ('full', 'x', 'y')}
    with original_modules(args.source_dir) as (rf, _, _):
        for source in recipe['sources']:
            record, m = source['record'], source['metadata']
            g = rf.geometry(m['width'], m['height'], record['targetRatioWH'])
            groups['full' if g is None else g[0]].append(record)
    if {k: len(v) for k, v in groups.items()} != recipe['geometry_counts']:
        raise ValueError('The complete frozen geometry partition differs')
    return groups


def commands(args):
    """Original twelve feature/head stages, two decodes and one final assembly."""
    root, src = args.out_dir, args.source_dir
    outputs, model = root / 'outputs', root / 'model_view'
    common = [sys.executable, '-B']
    def original(name, script, flags):
        if (getattr(args, 'decoder_policy', 'original') == 'capability'
                and script in ('tools/infer_yesno.py', 'tools/spatial_feats.py')):
            kind = 'temporal' if script == 'tools/infer_yesno.py' else 'spatial'
            argv = common + ['-m', 'aic.inference.checked_shared_script', '--kind', kind,
                             '--source-dir', str(src), '--recipe', str(args.recipe), '--', *map(str, flags)]
        else:
            argv = common + [str(src / script), *map(str, flags)]
        return {'name': name, 'argv': argv, 'cwd': str(src)}
    prob = outputs / 'original_yesno.probs.jsonl'
    feats = outputs / 'temporal_features'
    stages = [original('01_temporal_features', 'tools/infer_yesno.py', [
        '--model', model, '--index', args.index, '--video-dir', args.video_dir,
        '--out', prob, '--save-feats', feats, '--batch', 16, '--clip-sec', 3,
        '--clip-fps', 2, '--max-side', 448, '--max-window-sec', 32,
        '--feat-layers', '0.33,0.56,0.78,1.0'])]
    for layer in (1, 2):
        stages.append(original('02_temporal_head_L' + str(layer), 'tools/train_vlm_head.py', [
            '--predict', '--model-dir', args.heads / ('vh_restore_baseline_L' + str(layer)),
            '--probs', prob, '--feats', feats, '--channel', 'mix', '--device', 'cpu',
            '--out', outputs / ('L%d_mix.probs.jsonl' % layer)]))
    stages.append(original('03_temporal_ensemble', 'tools/ensemble_probs.py', [
        '--probs', outputs / 'L1_mix.probs.jsonl', outputs / 'L2_mix.probs.jsonl',
        '--weights', 1, 1, '--out', outputs / 'test.probs.jsonl']))
    for axis in ('x', 'y'):
        for side in (448, 672):
            batch = 8 if axis == 'x' and side == 672 else 16
            idx = root / ('index_' + axis + '.jsonl')
            feats = outputs / ('spatial_%s_%d' % (axis, side))
            stages.append(original('04_spatial_features_%s_%d' % (axis, side), 'tools/spatial_feats.py', [
                '--model', model, '--index', idx, '--video-dir', args.video_dir,
                '--out-dir', feats, '--no-generate', '--layers', '18,27', '--batch', batch, '--max-side', side]))
            head = ('sph_restore_%d_x' if axis == 'x' else 'sph_ybox3_%d') % side
            stages.append(original('05_spatial_head_%s_%d' % (axis, side), 'tools/train_spatial_head.py', [
                '--predict', '--model-dir', args.heads / head, '--feats', feats, '--index', idx,
                '--device', 'cpu', '--out', outputs / ('centers_%s_%d.jsonl' % (axis, side))]))
    for shift in (0, 3):
        stages.append(original('06_decode_shift' + str(shift), 'tools/decode_scores.py', [
            '--probs', outputs / 'test.probs.jsonl', '--params', root / ('shift%d_decode.json' % shift),
            '--index', args.index, '--video-dir', args.video_dir, '--safe-frames',
            '--write-submission', root / 'reconstructed_submission' / ('shift%d_frames.jsonl' % shift)]))
    flags = ['-m', 'aic.inference.shared_recipe', '--internal-assembly-root', str(root),
             '--recipe', str(args.recipe), '--source-dir', str(src), '--model', str(args.model),
             '--processor', str(args.processor), '--heads', str(args.heads), '--detector', str(args.detector),
             '--index', str(args.index), '--video-dir', str(args.video_dir), '--out-dir', str(root),
             '--max-seconds', str(args.max_seconds)]
    if args.addons:
        flags += ['--addons', str(args.addons)]
    stages.append({'name': '07_original_trim_placement_and_export', 'argv': common + flags,
                   'cwd': str(Path(__file__).resolve().parents[2])})
    if len(stages) != 15:
        raise ValueError('Complete15-stage graph required')
    return stages


def materialize(args, recipe, groups):
    root = args.out_dir
    for name, records in groups.items():
        with (root / ('index_' + name + '.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
    decode = json.loads((args.source_dir / 'tools/configs/d3_decode.json').read_bytes())
    if decode['alpha'] != .3:
        raise ValueError('Original alpha differs')
    for shift in (0, 3):
        write(root / ('shift%d_decode.json' % shift), dict(decode, shift=float(shift)))
    plan = {'kind': 'fixed_V8_raw_video_reproduction_command_graph', 'mode': args.mode,
            'execution_eligible': args.mode == 'run', 'neural_dispatch_in_plan_mode': False,
            'requested_maximum_seconds': args.max_seconds, 'original_deadline_unix': args.deadline_unix,
            'recipe_SHA256': sha(args.recipe), 'geometry_counts': recipe['geometry_counts'],
            'whole_pipeline_parameters': PARAMETERS, 'stages': commands(args),
            'decoder_policy': args.decoder_policy,
            'expected_prediction_SHA256': recipe['expected_prediction_sha256'],
            'new_training_or_method_search': False, 'automatic_submission': False,
            'full_neural_reproduction_verified': False}
    write(root / 'command_graph.json', plan)
    return plan


def feature_check(folder, ids, temporal, check):
    import numpy as np
    files = list(folder.glob('*.npz'))
    if len(files) != len(ids) or {p.stem for p in files} != set(ids):
        raise ValueError('Incomplete or duplicate feature population: ' + str(folder))
    entries = []
    for path in sorted(files):
        check()
        with np.load(path, allow_pickle=False) as data:
            keys = ('last', 'vis', 'margin', 'p_sec') if temporal else ('col', 'row')
            if any(not data[k].size or not np.isfinite(data[k]).all() for k in keys):
                raise ValueError('Missing or nonfinite feature: ' + path.name)
            if temporal:
                if (data['last'].shape != data['vis'].shape or data['vis'].shape != (len(data['p_sec']), 4, 4096)
                        or len(data['margin']) != len(data['p_sec'])):
                    raise ValueError('Original four-layer temporal feature shape differs')
            elif (data['col'].ndim != 4 or data['row'].ndim != 4
                  or data['col'].shape[1] != 2 or data['row'].shape[1] != 2
                  or data['col'].shape[-1] != 4096 or data['row'].shape[-1] != 4096
                  or not np.isnan(data['box']).all()):
                raise ValueError('Original no-generate spatial feature shape differs')
        entries.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path, check)})
    write(folder.parent / (folder.name + '_verification.json'), {'complete': True, 'files': entries})


def probability_check(path, ids):
    import numpy as np
    data = read_rows(path, ids)
    if any(not row.get('windows') or not len(row.get('p_sec', [])) or not np.isfinite(row['p_sec']).all() for row in data.values()):
        raise ValueError('Incomplete temporal outputs may not use a decoder fallback')


def average_centers(args, axis, ids):
    import numpy as np
    outputs = args.out_dir / 'outputs'
    rows = [read_rows(outputs / ('centers_%s_%d.jsonl' % (axis, side)), ids) for side in (448, 672)]
    with (outputs / ('centers_' + axis + '_average.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
        for vid, a in rows[0].items():
            a, b = a['center_' + axis], rows[1][vid]['center_' + axis]
            if not a or len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Incomplete two-resolution spatial centers')
            center = [round((x + y) / 2, 4) for x, y in zip(a, b)]
            stream.write(json.dumps({'video_id': vid, 'center_' + axis: center}) + '\n')


def model_view(args, recipe, check):
    """Create the same processor/model view; canonical assets remain read-only."""
    view = args.out_dir / 'model_view'
    view.mkdir()
    for entry in recipe['assets']['processor']:
        check()
        shutil.copyfile(args.processor / entry['path'], view / entry['path'])
    states = set()
    parameters = 0
    for entry in recipe['assets']['backbone']:
        path = args.model / entry['path']
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                if not 0 < length < 8 * 1024**2:
                    raise ValueError('Unexpected safetensors header bound')
                header = json.loads(stream.read(length))
            for name, spec in header.items():
                if name == '__metadata__':
                    continue
                if name in states or spec['dtype'] != 'BF16':
                    raise ValueError('Original TimeLens states differ')
                states.add(name)
                parameters += math.prod(spec['shape'])
            (view / path.name).symlink_to(path.resolve())
        else:
            shutil.copyfile(path, view / path.name)
    if len(states) != 750 or parameters != 8767123696 or (view / 'chat_template.jinja').exists():
        raise ValueError('Complete original750-state backbone and documented processor view required')
    from transformers import AutoConfig, AutoProcessor
    processor = AutoProcessor.from_pretrained(view, local_files_only=True)
    config = AutoConfig.from_pretrained(view, local_files_only=True)
    if config.text_config.num_hidden_layers != 36 or config.text_config.hidden_size != 4096 or 'do_resize' in processor.tokenizer.init_kwargs:
        raise ValueError('Original processor/model contract differs')
    write(args.out_dir / 'model_view_ready.json', {'complete': True, 'state_tensors': 750,
          'backbone_parameters': parameters, 'canonical_assets_modified': False})


def gpu_idle():
    value = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                           text=True, capture_output=True, timeout=15, check=True).stdout
    if value.strip():
        raise RuntimeError('A GPU process is live; do not overlap or stop it: ' + value.strip())
    return value


def resources(args):
    if args.jobs_root is None or not args.jobs_root.is_dir():
        raise ValueError('An explicit existing --jobs-root covering this workspace is required for run')
    for path in args.jobs_root.rglob('*experiment.json'):
        if path.resolve() == (args.out_dir / 'experiment.json').resolve():
            continue
        try:
            state = json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError('Cannot inspect a workspace job ledger: ' + str(path)) from exc
        if state.get('active'):
            proc = Path('/proc') / str(state.get('pid')) / 'cmdline'
            actual = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace') if proc.exists() else None
            raise RuntimeError('Inspect another active or stale ledger before dispatch: '
                               + str(path) + ' actual_command=' + repr(actual))
    return gpu_idle()


def reserve_gpu(jobs_root):
    path = jobs_root / '.aic_shared_V8_gpu.lock'
    nonce = uuid.uuid4().hex
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'nonce': nonce}, stream)
    return path, nonce


def release_gpu(reservation):
    if reservation is None:
        return
    path, nonce = reservation
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return
    if value.get('pid') == os.getpid() and value.get('nonce') == nonce:
        path.unlink()


def stop_own_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=3)
    except ProcessLookupError:
        pass


def stage_call(stage, env, end, log, child_ref):
    remaining = end - time.time()
    if remaining <= 0:
        raise TimeoutError('Original reproduction deadline reached')
    with Path(log).open('xb') as stream:
        child = subprocess.Popen(stage['argv'], cwd=stage['cwd'], env=env,
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        child_ref[0] = child
        try:
            code = child.wait(timeout=remaining)
            if code:
                raise RuntimeError('Stage failed, no automatic retry: ' + stage['name'] + ' exit=' + str(code))
        finally:
            stop_own_child(child)
            child_ref[0] = None


def assemble(args, recipe, check, root):
    """Only the active parent controller can invoke its final detector stage."""
    parent = json.loads((root / 'experiment.json').read_bytes())
    if (not parent['active'] or parent['pid'] != os.getppid() or len(parent['completed_stages']) != 14
            or parent['recipe_SHA256'] != sha(args.recipe)):
        raise ValueError('Assembly requires its actual live14-stage parent')
    end = parent['hard_deadline_unix']
    def within_parent():
        check()
        if time.time() >= end:
            raise TimeoutError('Parent reproduction deadline reached')
    import torch
    with original_modules(args.source_dir) as (rf, _, _):
        gpu_idle()
        if args.decoder_policy == 'capability':
            from aic.data.reliable_video import detector_reader
            rf.iter_frames = detector_reader(rf.iter_frames)
        detector = rf.Detector(str(args.detector), 'cuda:0', 640)
        if args.decoder_policy == 'capability':
            detector.tag += '+capability-decoder-v1'
        det_parameters = sum(p.numel() for p in detector.model.model.parameters())
        heads = sum(sum(t.numel() for t in torch.load(args.heads / entry['path'], map_location='cpu', weights_only=True).values())
                    for entry in recipe['assets']['heads'] if entry['path'].endswith('.pt'))
        if det_parameters != 9458752 or heads != 50701366 or 8767123696 + det_parameters + heads != PARAMETERS:
            raise ValueError('Actual complete inference parameter count differs')
        ids = {str(s['record']['video_id']) for s in recipe['sources']}
        decoded = read_rows(root / 'reconstructed_submission/shift0_frames.jsonl', ids)
        cache_entries = []
        cache_dir = root / 'reconstructed_submission/det_cache'
        config = json.loads((args.source_dir / 'tools/configs/final_AM.json').read_bytes())
        for source in recipe['sources']:
            within_parent()
            record, m = source['record'], source['metadata']
            vid = str(record['video_id'])
            if rf.geometry(m['width'], m['height'], record['targetRatioWH']) is None:
                continue
            frames = [p['frame'] for p in decoded[vid]['predictions']]
            if not frames:
                raise ValueError('Frozen V8 requires nonempty strip rows')
            path = str(args.video_dir / (vid + '.mp4'))
            data = rf.collect(path, frames, config['reframe']['stride'], detector, str(cache_dir))
            sampled = rf.sample_frames(frames, config['reframe']['stride'])
            if data['frames'] != sampled:
                raise ValueError('Actual detector sampling differs: ' + vid)
            path = Path(rf.cache_file(path, sampled, detector.tag, str(cache_dir)))
            cache_entries.append({'video_id': vid, 'path': path.relative_to(root).as_posix(),
                                  'bytes': path.stat().st_size, 'sha256': sha(path, within_parent),
                                  'detector': data['detector'], 'sampled_frames': len(sampled)})
    if len(cache_entries) != 254:
        raise ValueError('Complete254 detector inputs required')
    paths = {'shift0': 'reconstructed_submission/shift0_frames.jsonl',
             'shift3': 'reconstructed_submission/shift3_frames.jsonl',
             'centers_x': 'outputs/centers_x_average.jsonl', 'centers_y': 'outputs/centers_y_average.jsonl',
             'probabilities': 'outputs/test.probs.jsonl'}
    fresh = dict(recipe, detection_cache=cache_entries, cache_inputs={name: {
        'path': name_path, 'bytes': (root / name_path).stat().st_size, 'sha256': sha(root / name_path, within_parent)}
        for name, name_path in paths.items()})
    write(root / 'actual_neural_output_bindings.json', fresh)
    output = root / 'final'
    output.mkdir()
    result = replay(SimpleNamespace(cache_root=root, source_dir=args.source_dir, out_dir=output), fresh, within_parent)
    result.update(neural_reproduction_verified=True, actual_total_parameters=PARAMETERS,
                  official_score_verified_by_execution=False, automatic_submission=False)
    write(root / 'final_result.json', result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'run'), default='plan')
    for name in ('recipe', 'source-dir', 'model', 'processor', 'heads', 'detector', 'index', 'video-dir', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--addons', type=Path)
    p.add_argument('--jobs-root', type=Path, help='Explicit workspace-wide job-ledger root; required for run')
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--internal-assembly-root', type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('recipe', 'source_dir', 'model', 'processor', 'heads', 'detector', 'index', 'video_dir', 'out_dir', 'addons', 'jobs_root', 'internal_assembly_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    started = time.time()
    if not 1 <= args.max_seconds <= 21600 or args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        raise ValueError('Explicit bounded reproduction clock required')
    end = min(started + args.max_seconds, args.deadline_unix if args.deadline_unix is not None else float('inf'))
    def check():
        if time.time() >= end:
            raise TimeoutError('Original reproduction deadline reached')
    check()
    recipe = load_recipe(args.recipe)
    args.decoder_policy = recipe.get('fixed_settings', {}).get('decoder_policy', 'original')
    if args.decoder_policy not in ('original', 'capability'):
        raise ValueError('Unknown frozen decoder policy')
    source_code(args, recipe, check)
    if args.internal_assembly_root:
        assemble(args, recipe, check, args.internal_assembly_root)
        return
    if args.out_dir.exists():
        raise FileExistsError('A new reproduction directory is required')
    groups = partitions(args, recipe)
    check()
    if args.mode == 'run' and os.name != 'posix':
        raise RuntimeError('Full neural reproduction uses the verified Linux GPU environment')
    if args.mode == 'run':
        resources(args)
    args.out_dir.mkdir(parents=False)
    graph = materialize(args, recipe, groups)
    check()
    if args.mode == 'plan':
        write(args.out_dir / 'plan_result.json', {'complete': True, 'command_stages': len(graph['stages']),
              'geometry_counts': recipe['geometry_counts'], 'neural_dispatch': False,
              'full_neural_reproduction_verified': False, 'execution_eligible': False})
        print(json.dumps({'complete': True, 'mode': 'plan', 'stages': 15, 'neural_dispatch': False}))
        return
    child_ref, reservation, finished, lock = [None], [None], threading.Event(), threading.RLock()
    state = {'active': True, 'pid': os.getpid(), 'kind': 'fixed_V8_raw_video_reproduction',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'hard_deadline_unix': end,
             'recipe_SHA256': sha(args.recipe), 'completed_stages': [], 'training': False,
             'decoder_policy': args.decoder_policy,
             'submitted': False, 'neural_calls': None, 'full_neural_reproduction_verified': False}
    def save():
        with lock:
            target = args.out_dir / 'experiment.json'
            temp = target.with_suffix('.writing')
            temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            temp.replace(target)
    def guard():
        while not finished.wait(.25):
            if time.time() >= end:
                stop_own_child(child_ref[0])
                state.update(active=False, complete=False, original_hard_stop=True)
                save()
                write(args.out_dir / 'hard_stop_terminal.json', state)
                release_gpu(reservation[0])
                os._exit(124)
    error = None
    old_signals = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            old_signals[signum] = signal.signal(signum, interrupted)
        reservation[0] = reserve_gpu(args.jobs_root)
        save()
        threading.Thread(target=guard, daemon=True).start()
        resources(args)
        if args.addons:
            sys.path.insert(0, str(args.addons))
        qualification = prepare(args, recipe, check)
        for name, expected in recipe['neural_addon_versions'].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError('Original neural addon differs: ' + name)
        write(args.out_dir / 'preflight.json', qualification)
        check()
        model_view(args, recipe, check)
        for name in ('outputs', 'logs', 'reconstructed_submission'):
            (args.out_dir / name).mkdir()
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(args.addons) if args.addons else None,
                    str(Path(__file__).resolve().parents[2]), str(args.source_dir), str(args.source_dir / 'tools')])),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'YOLO_AUTOINSTALL': 'false', 'YOLO_OFFLINE': 'true', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
        all_ids = {str(s['record']['video_id']) for s in recipe['sources']}
        for stage in graph['stages']:
            check()
            state['current_stage'] = stage['name']
            save()
            if stage['name'].startswith(('01_', '04_')):
                resources(args)
            call = dict(stage, argv=list(stage['argv']))
            if stage['name'].startswith('07_'):
                call['argv'] += ['--deadline-unix', str(end)]
            stage_call(call, env, end, args.out_dir / 'logs' / (stage['name'] + '.log'), child_ref)
            name = stage['name']
            output = args.out_dir / 'outputs'
            if name == '01_temporal_features':
                probability_check(output / 'original_yesno.probs.jsonl', all_ids)
                feature_check(output / 'temporal_features', all_ids, True, check)
            elif name.startswith('02_'):
                probability_check(output / (name[-2:] + '_mix.probs.jsonl'), all_ids)
            elif name == '03_temporal_ensemble':
                probability_check(output / 'test.probs.jsonl', all_ids)
            elif name.startswith('04_'):
                axis, side = name.split('_')[-2:]
                feature_check(output / ('spatial_' + axis + '_' + side), {str(x['video_id']) for x in groups[axis]}, False, check)
            elif name.startswith('05_') and name.endswith('_672'):
                axis = name.split('_')[-2]
                average_centers(args, axis, {str(x['video_id']) for x in groups[axis]})
            elif name.startswith('06_'):
                read_rows(args.out_dir / 'reconstructed_submission' / (name[-6:] + '_frames.jsonl'), all_ids)
            state['completed_stages'].append(name)
            save()
        result = json.loads((args.out_dir / 'final_result.json').read_bytes())
        check()
        if not result['complete_payload_matches_release'] or not result['neural_reproduction_verified']:
            raise ValueError('Complete raw-neural reproduction not proven')
        state['full_neural_reproduction_verified'] = True
    except BaseException as exc:
        stop_own_child(child_ref[0])
        error = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        raise
    finally:
        finished.set()
        state.update(active=False, complete=error is None, error=error,
                     ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started)
        save()
        write(args.out_dir / 'terminal.json', state)
        release_gpu(reservation[0])
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
