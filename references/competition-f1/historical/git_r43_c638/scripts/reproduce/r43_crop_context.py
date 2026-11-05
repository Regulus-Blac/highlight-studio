"""Task-aligned y training: rendered-crop IoU and five-second frozen-feature context.

Three fixed arms, original source-group five-fold CV. This is reused-corpus
research validation, not a new blind holdout or an official competition score.
"""
import argparse
import gc
import hashlib
import itertools
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from r42_yopt import ROOT, ensemble, sh, write

OUT = ROOT / 'outputs/experiments/sym43_crop_context'
ARMS = ['baseline', 'crop_iou', 'context_iou']


class ContextHead(sh.SpatialHead):
    def __init__(self, D=4096):
        super().__init__(D)
        self.context_norm = nn.LayerNorm(D)
        self.context_proj = nn.Sequential(nn.Linear(D * 2, 64), nn.GELU())
        self.context_conv = nn.Sequential(nn.Conv1d(64, 64, 3), nn.GELU(), nn.Conv1d(64, 1, 3))
        nn.init.zeros_(self.context_conv[-1].weight)
        nn.init.zeros_(self.context_conv[-1].bias)

    def forward(self, x, pos, box, context):
        center = super().forward(x, pos, box)
        z = self.context_proj(self.context_norm(context).flatten(-2)).transpose(1, 2)
        delta = .1 * self.context_conv(z).flatten().tanh()
        return center + delta


def summaries(prof):
    pos = (np.arange(prof.shape[1], dtype=np.float32) + .5) / prof.shape[1] - .5
    return np.stack([prof.mean(1), (prof * pos[None, :, None]).mean(1)], 1)


def context_batch(batch, device):
    a = [b[4][np.clip(np.arange(b[5] - 2, b[5] + 3), 0, len(b[4]) - 1)] for b in batch]
    return torch.from_numpy(np.stack(a)).to(device)


def load(rows, meta, side):
    data = {}
    for r in rows:
        v = r['video_id']; m = meta[v]
        prof, zs = sh.load_clip(str(ROOT / ('outputs/spf_yhead_%d' % side) / (v + '.npz')), 'y', 0)
        y = np.array([p['center'][1] for p in r['teacher_signals']['trajectory']], np.float32)
        assert len(prof) == len(y)
        width = min(m['width'], int(m['height'] * 16 // 9))
        h = width * 9 / 16 / m['height']
        data[v] = dict(prof=prof, zs=zs, y=y, h=h, safe_h=np.ceil(width*9/16)/m['height'], height=m['height'], ctx=summaries(prof),
                       group=r['provenance']['source_group'])
    print('LOADED', side, len(data), flush=True)
    return data


def loss_value(pred, target, extent, safe_extent, arm):
    if arm == 'baseline':
        return (pred - target).abs().mean()
    lo, hi = safe_extent / 2, 1 - safe_extent / 2
    a = torch.minimum(torch.maximum(pred, lo), hi)
    b = torch.minimum(torch.maximum(target, lo), hi)
    delta = (a - b).abs()
    iou = (extent - delta).clamp_min(0) / (extent + delta)
    # The auxiliary term retains gradients for disjoint crops and saturated boundaries.
    return (1 - iou + .2 * (pred - target).abs() / extent).mean()


def fit(data, ids, arm, seed, dest, epochs=15):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = (ContextHead() if arm == 'context_iou' else sh.SpatialHead(4096)).cuda()
    if dest.exists():
        # Completed original-L1 folds are unaffected by the geometry correction.
        cfg=json.loads((dest/'config.json').read_text())
        assert arm=='baseline' and cfg['arm']=='baseline' and cfg['training_video_ids']==ids
        assert cfg['seed']==seed and cfg['epochs']==epochs
        model.load_state_dict(torch.load(dest/'head.pt',map_location='cuda',weights_only=True))
        print('REUSE_COMPLETED_ORIGINAL_FOLD',dest.name,flush=True)
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    items = [(data[v]['prof'][s], data[v]['zs'][s], data[v]['y'][s], data[v]['h'], data[v]['ctx'], s,data[v]['safe_h'])
             for v in ids for s in range(len(data[v]['y']))]
    start = time.time()
    for epoch in range(epochs):
        model.train(); total = 0.
        for batch in sh.batches(items, 128, rng):
            X, P, B, y = sh.to_t([b[:3] for b in batch], 'cuda')
            h = torch.tensor([b[3] for b in batch], dtype=torch.float32, device='cuda')
            safe_h = torch.tensor([b[6] for b in batch], dtype=torch.float32, device='cuda')
            pred = model(X, P, B, context_batch(batch, 'cuda')) if arm == 'context_iou' else model(X, P, B)
            loss = loss_value(pred, y, h, safe_h, arm)
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('TRAIN', dest.name, arm, epoch + 1, round(total / len(items), 6), 'sec', round(time.time()-start, 1), flush=True)
    dest.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), dest / 'head.pt')
    write(dest / 'config.json', {'D':4096, 'axis':'y', 'layer_idx':0, 'arm':arm, 'context_radius':2 if arm=='context_iou' else 0,
          'context_residual_bound':.1, 'loss_aux_l1_over_crop_extent':.2, 'seed':seed, 'epochs':epochs, 'batch':128,
          'lr':5e-4,'weight_decay':.05,'training_video_ids':ids,'training_seconds':len(items),
          'parameters':sum(p.numel() for p in model.parameters())})
    return model


@torch.no_grad()
def predict(model, d, arm, device='cuda'):
    model.eval(); result=[]
    for first in range(0,len(d['prof']),128):
        batch=[(d['prof'][s],d['zs'][s],0.,0.,d['ctx'],s) for s in range(first,min(first+128,len(d['prof'])))]
        X,P,B,_=sh.to_t([b[:3] for b in batch],device)
        z=model(X,P,B,context_batch(batch,device)) if arm=='context_iou' else model(X,P,B)
        result.extend(z.cpu().numpy().astype(float).tolist())
    assert np.isfinite(result).all()
    return np.clip(result,0,1).tolist()


def rendered_iou(pred,target,m):
    W,H=m['width'],m['height'];w=min(W,H*16//9);h=w*9/16
    safe_h=np.ceil(h)
    def top(c):
        return np.rint(np.clip(c*H-safe_h/2,0,H-safe_h))
    delta=np.abs(top(np.asarray(pred))-top(np.asarray(target)))
    return np.maximum(h-delta,0)/(h+delta)


def score(rows,meta,a,b):
    return {r['video_id']:float(rendered_iou(ensemble(a[r['video_id']],b[r['video_id']]),
             [p['center'][1] for p in r['teacher_signals']['trajectory']],meta[r['video_id']]).mean()) for r in rows}


def compare(rows,base,cand):
    groups={}
    for r in rows:groups.setdefault(r['provenance']['source_group'],[]).append(cand[r['video_id']]-base[r['video_id']])
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    index=np.random.default_rng(43).integers(0,len(groups),(10000,len(groups)))
    delta=sums[index].sum(1)/counts[index].sum(1)
    return {'baseline_crop_iou':float(np.mean(list(base.values()))),'candidate_crop_iou':float(np.mean(list(cand.values()))),
            'delta':float(np.mean([cand[v]-base[v] for v in base])),
            'bonferroni8_two_sided_99_375ci':np.quantile(delta,[.003125,.996875]).tolist()}


def run(rows,meta):
    groups=sorted({r['provenance']['source_group'] for r in rows});random.Random(0).shuffle(groups)
    folds={g:i%5 for i,g in enumerate(groups)}
    write(OUT/'protocol.json',{'arms':ARMS,'fold_by_source':folds,'fold_seeds':list(range(5)),'final_seed':99,'epochs':15,
          'primary_metric':'macro per-video rendered max-size 16:9 crop IoU to provided teacher centers',
          'validation_status':'source-group OOF on previously used corpus, not new blind or official test',
          'selection':'all 9 fixed 448/672 pairs; require familywise-adjusted positive lower CI to endorse',
          'export':'full-data point-best pair is experimental unless adjusted CI passes; baseline preserved',
          'test_training':False,'label_sha256':hashlib.sha256((ROOT/'data/ybox_train_v3.jsonl').read_bytes()).hexdigest(),
          'geometry_sha256':hashlib.sha256((ROOT/'data/ybox_geometry.jsonl').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    oof={side:{} for side in [448,672]}
    for side in [448,672]:
        data=load(rows,meta,side);ids=sorted(data)
        for arm in ARMS:
            pred={}
            for fold in range(5):
                train=[v for v in ids if folds[data[v]['group']]!=fold]; val=[v for v in ids if folds[data[v]['group']]==fold]
                model=fit(data,train,arm,fold,OUT/('cv_%s_%d_%d'%(arm,side,fold)))
                pred.update({v:predict(model,data[v],arm) for v in val});del model
            oof[side][arm]=pred;write(OUT/('oof_%s_%d.json'%(arm,side)),pred)
        del data;gc.collect()
    base=score(rows,meta,oof[448]['baseline'],oof[672]['baseline']);comparisons={}
    for a,b in itertools.product(ARMS,repeat=2):comparisons[a+'+'+b]=compare(rows,base,score(rows,meta,oof[448][a],oof[672][b]))
    chosen=max(comparisons,key=lambda name:comparisons[name]['candidate_crop_iou']);arms=dict(zip(['448','672'],chosen.split('+')))
    endorsed=chosen!='baseline+baseline' and comparisons[chosen]['bonferroni8_two_sided_99_375ci'][0]>0
    result={'candidate_arms':arms,'cv_supported':bool(endorsed),'selection':chosen,'comparisons':comparisons,
            'limitation':'reused-corpus teacher crop IoU proxy; no official score or new blind validation'}
    write(OUT/'selection.json',result);print('SELECTION',json.dumps(result),flush=True)
    for side in [448,672]:
        arm=arms[str(side)];target=OUT/('candidate_%d'%side)
        if arm=='baseline':
            shutil.copytree(ROOT/('outputs/experiments/sym42_yopt/final_%d'%side),target)
        else:
            data=load(rows,meta,side);model=fit(data,sorted(data),arm,99,target);del model,data;gc.collect()
    write(OUT/'training_complete.json',dict(result,complete=True,
          head_sha256={str(s):hashlib.sha256((OUT/('candidate_%d'%s)/'head.pt').read_bytes()).hexdigest() for s in [448,672]}))
    print('TRAINING_COMPLETE',chosen,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume-original-folds',action='store_true');args=ap.parse_args()
    torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in (ROOT/'data/ybox_train_v3.jsonl').read_text().splitlines()]
    meta={r['video_id']:r for r in map(json.loads,(ROOT/'data/ybox_geometry.jsonl').read_text().splitlines())}
    assert len(rows)==len(meta)==1516
    if args.smoke:
        d=load(rows[:2],meta,448)
        for arm in ['crop_iou']:
            model=fit(d,sorted(d),arm,0,OUT/('smoke_exact_geometry_'+arm),1)
            for v in d:assert len(predict(model,d[v],arm))==len(d[v]['y'])
        write(OUT/'smoke_complete.json',{'passed':True})
    else:
        if (OUT/'protocol.json').exists():
            assert args.resume_original_folds
            (OUT/'protocol.json').rename(OUT/'protocol_before_safe_height_correction.json')
        run(rows,meta)
