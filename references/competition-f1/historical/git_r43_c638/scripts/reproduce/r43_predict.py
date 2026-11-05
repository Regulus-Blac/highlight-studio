"""Reload baseline or temporal-context y weights and infer exact cached seconds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from r43_crop_context import ContextHead,predict,sh,summaries

p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--feats',type=Path,required=True)
p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
p.add_argument('--device',default='cpu')
a=p.parse_args();torch.set_num_threads(8)
cfg=json.loads((a.model_dir/'config.json').read_text());arm=cfg.get('arm','baseline')
model=(ContextHead() if arm=='context_iou' else sh.SpatialHead(4096)).to(a.device)
model.load_state_dict(torch.load(a.model_dir/'head.pt',map_location=a.device,weights_only=True))
rows=[json.loads(l) for l in a.index.read_text().splitlines() if l.strip()]
if a.limit:rows=rows[:a.limit]
with a.out.open('x') as f:
    for r in rows:
        v=str(r['video_id']);prof,zs=sh.load_clip(str(a.feats/(v+'.npz')),'y',0)
        raw=(predict(model,dict(prof=prof,zs=zs,ctx=summaries(prof)),arm,device=a.device) if arm=='context_iou'
             else sh.predict(model,prof,zs,a.device))
        c=sh.medfilt(raw)
        assert np.isfinite(c).all()
        f.write(json.dumps({'video_id':v,'center_y':[round(float(x),4) for x in c]})+'\n')
print('PREDICTED',len(rows),a.out)
