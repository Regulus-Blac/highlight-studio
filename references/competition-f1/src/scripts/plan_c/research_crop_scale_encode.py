#!/usr/bin/env python3
"""Encode each resolution on its raw-cache host; combine without changing the frozen learner."""
import argparse
import hashlib
import json
import platform
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

import research_crop_scale as original


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_at(data):
    return [r for split in ('train', 'validation')
            for r in map(json.loads, (data / (split + '_clean.jsonl')).open())]


def difference(a, b):
    return dict(byteexact=bool(np.array_equal(a, b)), max_abs=float(np.max(np.abs(a - b))),
                mean_abs=float(np.mean(np.abs(a - b))))


def smoke_comparison(ids, features, base, triples, target, smoke, res):
    reference = np.load(smoke / 'smoke.npz')
    rr = list(map(json.loads, (smoke / 'smoke_rows.jsonl').open()))
    lookup = {v: i for i, v in enumerate(ids)}
    chosen = [lookup[r['video_id']] for r in rr]
    assert np.array_equal(base[chosen], reference['base'])
    assert np.array_equal(triples[chosen], reference['triplets'])
    assert np.array_equal(target[chosen], reference['target'])
    expected = reference['features'] if res is None else reference['features'][:, (448, 672).index(res)]
    return dict(images=len(chosen), geometry_and_target_exact=True,
                encoded_features=difference(features[chosen], expected))


def encode(args):
    started = time.time()
    rows = rows_at(args.data)
    assert len(rows) == 5223
    coverage = json.loads(args.coverage.read_text())
    assert coverage['status'] == 'COMPLETE_VERIFIED'
    if args.res == 448:
        expected = {v: info['sha256'] for v, info in coverage['coverage'].items()}
    else:
        expected = {Path(p).stem: digest for p, digest in coverage['files_sha256'].items()}
    assert set(expected) == {r['video_id'] for r in rows}
    models, model_inputs = original.load_heads(args.g1, args.x1)
    plan = json.loads(args.plan.read_text())
    expected_core = next(v for p,v in plan['source_sha256'].items() if p.endswith('/scripts/plan_c/research_crop_scale.py'))
    assert sha(original.__file__) == expected_core
    for name, entry in plan['frozen_heads'].items():
        axis = 'x' if '/x1/' in name else 'y'
        root = args.x1 if axis == 'x' else args.g1
        path = root / Path(name).parent.name / Path(name).name
        assert sha(path) == entry['sha256'], path
    features, bases, triples, raw = [], [], [], {}
    for number, row in enumerate(rows, 1):
        assert row['targetRatioWH'] == [16, 9]
        assert original.geometry(row['width'], row['height'], row['targetRatioWH']) is None
        path = args.features / f'feats_{args.res}' / (row['video_id'] + '.npz')
        digest = sha(path)
        assert digest == expected[row['video_id']], path
        assert np.array_equal(np.load(path)['frame_indices'], [0])
        vectors = []
        for axis in ('x', 'y'):
            model, cfg = models[args.res, axis]
            profile, zs = original.load_clip(path, axis, cfg['layer_idx'])
            encoded, _ = original.encode(model, profile, zs)
            vectors.append(encoded[0])
        box = original.baseline_box(row, {})
        features.append(np.r_[*vectors, original.geo_features(row, box)])
        bases.append(original.normalized_box(box, row))
        triples.append(box)
        raw[str(path)] = digest
        if number == 2:
            first = smoke_comparison([r['video_id'] for r in rows[:2]],np.array(features,np.float32),
                np.array(bases,np.float32),np.array(triples),
                np.array([r['original_crop_xyxy'] for r in rows[:2]],np.float32),args.smoke,args.res)
            print('CROP_SCALE_ENCODE_TWO_REAL_FULL_SMOKE',args.res,json.dumps(first),flush=True)
        if number % 100 == 0:
            print('CROP_SCALE_ENCODE', args.res, number, len(rows), round(time.time() - started, 1), flush=True)
    features, bases, triples = np.array(features, np.float32), np.array(bases, np.float32), np.array(triples)
    target = np.array([r['original_crop_xyxy'] for r in rows], np.float32)
    ids = np.array([r['video_id'] for r in rows])
    assert features.shape == (5223, 1550) and np.isfinite(features).all()
    comparison = smoke_comparison(ids, features, bases, triples, target, args.smoke, args.res)
    output = args.out / f'encoded_{args.res}.npz'
    np.savez(output, ids=ids, features=features, base=bases, triplets=triples, target=target)
    raw_path = args.out / f'encoded_{args.res}_raw_sha256.json'
    raw_path.write_text(json.dumps(raw, indent=2) + '\n')
    report = dict(status='COMPLETE_VERIFIED',resolution=args.res,images=len(rows),feature_shape=list(features.shape),
                  seconds=time.time() - started,threads=4,gpu_used=False,validation_scored=False,
                  smoke_comparison=comparison,environment=dict(torch=torch.__version__,numpy=np.__version__,
                      python=platform.python_version(),machine=platform.machine(),processor=platform.processor()),
                  inputs_sha256={str(p):sha(p) for p in [Path(__file__),Path(original.__file__),args.coverage,args.plan,
                      args.data/'train_clean.jsonl',args.data/'validation_clean.jsonl',Path('tools/train_spatial_head.py'),
                      Path('tools/reframe.py'),Path('highlight/geometry.py')]},frozen_model_sha256=model_inputs,
                  output_sha256={str(p):sha(p) for p in [output,raw_path]},
                  original_core_model_and_hyperparameters_unchanged=True,
                  numeric_policy='Fixed448 host and672 host; record real CPU differences, never replace values by smoke/reference outputs.')
    report_path = args.out / f'encoded_{args.res}.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    members = [output,raw_path,report_path]
    manifest = args.out / f'encoded_{args.res}_members.json'
    manifest.write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in members},indent=2)+'\n')
    archive = args.out / f'encoded_{args.res}.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in members+[manifest]:tar.add(path,arcname=path.name,recursive=False)
    receipt = dict(status='COMPLETE_VERIFIED',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
                   member_manifest_sha256=sha(manifest),members=len(members)+1)
    (args.out / f'encoded_{args.res}_archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('CROP_SCALE_ENCODE_COMPLETE',json.dumps(receipt),flush=True)


def combine(args):
    rows = rows_at(args.data)
    ids = [r['video_id'] for r in rows]
    views = {}
    for res in (448,672):
        report_path = args.out/f'encoded_{res}.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'COMPLETE_VERIFIED'
        path = args.out/f'encoded_{res}.npz'
        expected = next(d for p,d in report['output_sha256'].items() if p.endswith(f'encoded_{res}.npz'))
        assert sha(path) == expected
        z = np.load(path)
        lookup = {v:i for i,v in enumerate(z['ids'].tolist())}
        assert len(lookup) == len(ids) and set(lookup) == set(ids)
        order = [lookup[v] for v in ids]
        views[res] = {k:z[k][order] for k in ('features','base','triplets','target')}
    for key in ('base','triplets','target'):
        assert np.array_equal(views[448][key],views[672][key]),key
    assert np.array_equal(views[448]['target'],np.array([r['original_crop_xyxy'] for r in rows],np.float32))
    features = np.stack([views[res]['features'] for res in (448,672)],1)
    reference = smoke_comparison(ids,features,views[448]['base'],views[448]['triplets'],views[448]['target'],args.smoke,None)
    args.fit.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation'):
        keep = np.array([r['dataset_split'] == split for r in rows])
        selected = [r for r,k in zip(rows,keep) if k]
        np.savez(args.fit/(split+'.npz'),features=features[keep],
                 **{k:views[448][k][keep] for k in ('base','triplets','target')})
        (args.fit/(split+'_rows.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in selected))
    groups = [{r['provenance']['source_group'] for r in rows if r['dataset_split']==split} for split in ('train','validation')]
    assert not groups[0]&groups[1]
    report = dict(status='COMPLETE_VERIFIED',images=len(rows),train=4504,validation=719,
                  shape=list(features.shape),ids_base_triplets_target_exact=True,smoke_comparison=reference,
                  original_train_evaluate_code_unchanged=True,validation_scored=False,
                  inputs_sha256={str(p):sha(p) for p in [Path(__file__),Path(original.__file__),args.plan,
                      args.out/'encoded_448.json',args.out/'encoded_672.json']},
                  output_sha256={str(p):sha(p) for p in args.fit.iterdir() if p.is_file()})
    (args.fit/'combine_verified.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CROP_SCALE_COMBINE_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','output_sha256')}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['encode','combine'],required=True)
    ap.add_argument('--res',type=int,choices=[448,672])
    ap.add_argument('--coverage',type=Path)
    ap.add_argument('--data',type=Path,default=Path('outputs/research_44_76/learned_box/full'))
    ap.add_argument('--features',type=Path,default=Path('outputs/research_44_76/learned_box/features'))
    ap.add_argument('--g1',type=Path,default=Path('outputs/research_44_76/crop_scale/frozen_heads/g1'))
    ap.add_argument('--x1',type=Path,default=Path('outputs/research_44_76/crop_scale/frozen_heads/x1'))
    ap.add_argument('--plan',type=Path,default=Path('outputs/research_44_76/crop_scale/plan.json'))
    ap.add_argument('--smoke',type=Path,default=Path('outputs/research_44_76/crop_scale/smoke'))
    ap.add_argument('--out',type=Path,default=Path('outputs/research_44_76/crop_scale/encoded'))
    ap.add_argument('--fit',type=Path,default=Path('outputs/research_44_76/crop_scale/fit'))
    args=ap.parse_args();torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=True)
    encode(args) if args.mode=='encode' else combine(args)


if __name__=='__main__':main()
