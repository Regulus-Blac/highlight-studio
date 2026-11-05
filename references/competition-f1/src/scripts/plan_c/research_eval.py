#!/usr/bin/env python3
"""Paired source-group bootstrap; development results never open the blind gate."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import decode_scores as ds


def paired(a,b,groups):
    ids=sorted(a)
    assert set(ids)==set(b), 'paired comparison requires identical videos'
    by_group=defaultdict(list)
    for v in ids:by_group[groups[v]].append(v)
    blocks=list(by_group.values());rng=np.random.default_rng(20261022)
    boot=[]
    for _ in range(5000):
        sample=[v for i in rng.integers(len(blocks),size=len(blocks)) for v in blocks[i]]
        boot.append(np.mean([b[v]-a[v] for v in sample]))
    return dict(videos=len(ids),sources=len(blocks),baseline=float(np.mean(list(a.values()))),
                candidate=float(np.mean(list(b.values()))),delta=float(np.mean([b[v]-a[v] for v in ids])),
                ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--gt',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['temporal','spatial'],required=True)
    ap.add_argument('--params',default='tools/configs/d3_decode.json')
    ap.add_argument('--split',choices=['dev','val'],default='dev')
    args=ap.parse_args()
    rows={r['video_id']:r for r in map(json.loads,open(args.gt)) if r['dataset_split']==args.split}
    groups={v:r['provenance']['source_group'] for v,r in rows.items()}
    results=[]
    for filename in (args.baseline,args.candidate):
        pred={r['video_id']:r for r in map(json.loads,open(filename))}
        assert set(rows)<=set(pred), 'missing evaluated videos'
        if args.mode=='temporal':
            videos=ds.load_val_videos(args.gt,pred,'all',include_empty=True)
            params=dict(ds.DEFAULT_PARAMS,**json.load(open(args.params)))
            results.append({v:x[0] for v,x in ds.evaluate(videos,pred,params,list(rows)).items()})
        else:
            results.append({v:pred[v]['iou'] for v in rows})
    report=paired(*results,groups)
    report.update(mode=args.mode,metric='temporal_F' if args.mode=='temporal' else 'teacher_crop_IoU',split=args.split,human_gate_passed=False,
                  files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.baseline,args.candidate,args.gt,args.params]})
    Path(args.out).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
