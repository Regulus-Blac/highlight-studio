"""Combine capability-corrected shared inference with trained y heads and absolute full-frame scales.

New unscored experiments; no bypass of an old release's expected-SHA contract.
No test reference labels, manual boxes, online teacher or automatic submission.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from r43_crop_context import ROOT,OUT,write
from r43_readers import COT,reliable

sys.path.insert(0,str(COT))
from aic.deployment import shared_reproduction as shared
from aic.inference.shared_recipe import feature_check,probability_check

T=Path('/root/sj-fs/autodl-tmp/public/aic-competition-test/2026_semifinal')
RUN=ROOT/'outputs/experiments/sym43_hybrid'
PY=sys.executable


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def call(name,argv):
    print('STAGE',name,flush=True)
    with (RUN/'logs'/(name+'.log')).open('w') as log:
        subprocess.run([PY,*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
          env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})


def records(p):
    rows=[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    result={str(r['video_id']):r for r in rows};assert len(result)==len(rows)
    return result


def merged_features(native,corrected,destination,ids,affected):
    destination.mkdir()
    for v in ids:
        p=(corrected if v in affected else native)/(v+'.npz')
        assert p.is_file()
        (destination/p.name).symlink_to(p.resolve())
    return destination


def average(paths,key,dest):
    a,b=map(records,paths);assert set(a)==set(b)
    with dest.open('x') as f:
        for v in a:
            c=(np.asarray(a[v][key])+np.asarray(b[v][key]))/2
            f.write(json.dumps({'video_id':v,key:[round(float(x),4) for x in c]})+'\n')


def model_view():
    """Use the documented Qwen processor, without the extra TimeLens Jinja override."""
    import struct
    from transformers import AutoConfig,AutoProcessor
    canonical=Path('/root/sj-fs/models/TimeLens-8B');view=RUN/'model_view';view.mkdir()
    binding=json.loads((ROOT/'data/qwen_processor_0c351dd.json').read_text())
    for entry in binding['files']:
        source=canonical/entry['path']
        assert source.stat().st_size==entry['bytes'] and digest(source)==entry['sha256']
        shutil.copyfile(source,view/entry['path'])
    for name in ['config.json','generation_config.json','model.safetensors.index.json']:
        shutil.copyfile(canonical/name,view/name)
    states=set();parameters=0
    for name in sorted(set(json.loads((view/'model.safetensors.index.json').read_text())['weight_map'].values())):
        source=canonical/name
        with source.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];assert 0<n<8*1024**2
            header=json.loads(f.read(n))
        for key,spec in header.items():
            if key=='__metadata__':continue
            assert key not in states and spec['dtype']=='BF16';states.add(key)
            parameters+=int(np.prod(spec['shape']))
        (view/name).symlink_to(source.resolve())
    assert len(states)==750 and parameters==8767123696 and not (view/'chat_template.jinja').exists()
    proc=AutoProcessor.from_pretrained(view,local_files_only=True);cfg=AutoConfig.from_pretrained(view,local_files_only=True)
    assert cfg.text_config.num_hidden_layers==36 and cfg.text_config.hidden_size==4096 and 'do_resize' not in proc.tokenizer.init_kwargs
    write(RUN/'model_view_ready.json',dict(complete=True,state_tensors=len(states),backbone_parameters=parameters,
          processor_revision=binding['revision'],files=binding['files'],canonical_assets_modified=False))
    return view


def package(path,index):
    from fractions import Fraction
    md=json.loads((OUT/'decoder_metadata.json').read_text())
    actual=records(path);assert len(actual)==426
    for r in actual.values():
        m=md[str(r['video_id'])];tw,th=map(Fraction,r['targetRatioWH']);last=-1
        for p in r['predictions']:
            f=p['frame'];x,y,w=p['bboxes']
            assert type(f) is int and last<f<m['frames'] and all(type(v) is int for v in [x,y,w])
            assert x>=0 and y>=0 and w>0 and x+w<=m['width'] and y*tw+w*th<=m['height']*tw
            last=f
    call(path.stem+'_validate',['tools/validate_submission.py','--pred',path,'--index',T/'submission_index.jsonl','--video-dir',T/'video'])
    out=path.with_suffix('.zip')
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:z.write(path,'submission.jsonl')
    with zipfile.ZipFile(out) as z:assert z.namelist()==['submission.jsonl'] and z.testzip() is None
    return {'jsonl':str(path),'sha256':digest(path),'zip':str(out),'zip_sha256':digest(out),
            'videos':len(index),'frames':sum(len(r['predictions']) for r in records(path).values()),'official_score':None}


def main():
    assert (OUT/'training_complete.json').exists()
    qualification=json.loads((OUT/'decoder_qualification.json').read_text());assert qualification['complete']
    RUN.mkdir(exist_ok=False);(RUN/'logs').mkdir();(RUN/'features').mkdir();(RUN/'corrections').mkdir();(RUN/'cache').mkdir()
    rows=[json.loads(l) for l in (T/'submission_index.jsonl').read_text().splitlines() if l.strip()]
    metadata=json.loads((OUT/'decoder_metadata.json').read_text());ids=[str(r['video_id']) for r in rows];assert len(set(ids))==426
    # Bind actual inputs and actual source code for this experiment, not a claimed release identity.
    sources=[{'record':r,'metadata':metadata[str(r['video_id'])],'sha256':digest(T/'video'/(str(r['video_id'])+'.mp4'))} for r in rows]
    upstream=[*ROOT.joinpath('tools').glob('*.py'),*ROOT.joinpath('highlight').glob('*.py'),*ROOT.joinpath('baseline').glob('*.py')]
    write(RUN/'extractor_binding.json',{'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in upstream},
          'sym_commit':'8a18db2ef5dd6bf0d8a2eb88a274341421479a43','cotsheep_commit':'a4809a1'})
    affected={v for v,m in metadata.items() if not m['opencv_transfer_supported']}
    assert affected==set(qualification['sources'])
    for v in affected:assert digest(T/'video'/(v+'.mp4'))==qualification['sources'][v]['source_sha256']
    with shared.original_modules(ROOT) as (rf,tf,geometry):
        groups={k:[r for r in rows if (('full' if (g:=rf.geometry(metadata[str(r['video_id'])]['width'],metadata[str(r['video_id'])]['height'],r['targetRatioWH'])) is None else g[0])==k)] for k in ['full','x','y']}
    assert {k:len(v) for k,v in groups.items()}=={'full':172,'x':123,'y':131}
    for k,rr in groups.items():(RUN/('index_'+k+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rr))
    corrected=[r for r in rows if str(r['video_id']) in affected]
    (RUN/'index_affected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in corrected))
    model=model_view()
    temporal=RUN/'corrections/temporal'
    if affected:
        call('correct_temporal',['scripts/reproduce/r43_readers.py','--kind','temporal','--source-dir',ROOT,
          '--recipe',RUN/'extractor_binding.json','--','--model',model,'--index',RUN/'index_affected.jsonl','--video-dir',T/'video',
          '--out',RUN/'corrections/raw.probs.jsonl','--save-feats',temporal,'--batch','16'])
        feature_check(temporal,affected,True,lambda:None)
    raw=records(ROOT/'outputs/rz_test_timelens8b.probs.jsonl')
    if affected:raw.update(records(RUN/'corrections/raw.probs.jsonl'))
    assert set(raw)==set(ids)
    raw_path=RUN/'raw.probs.jsonl';raw_path.write_text(''.join(json.dumps(raw[v])+'\n' for v in ids))
    merged=merged_features(ROOT/'outputs/rzf_test_timelens8b',temporal,RUN/'features/temporal',ids,affected)
    for layer in [1,2]:
        call('time_head_%d'%layer,['tools/train_vlm_head.py','--predict','--model-dir',ROOT/('outputs/vh_restore_baseline_L%d'%layer),
          '--probs',raw_path,'--feats',merged,'--channel','mix','--device','cpu','--out',RUN/('L%d.probs.jsonl'%layer)])
        probability_check(RUN/('L%d.probs.jsonl'%layer),set(ids))
    probs=RUN/'test.probs.jsonl'
    call('time_ensemble',['tools/ensemble_probs.py','--probs',RUN/'L1.probs.jsonl',RUN/'L2.probs.jsonl','--out',probs])
    probability_check(probs,set(ids))
    centers={}
    for axis in ['x','y']:
        axis_ids=[str(r['video_id']) for r in groups[axis]];changed=set(axis_ids)&affected
        for side in [448,672]:
            if changed:
                idx=RUN/('index_affected_'+axis+'.jsonl');idx.write_text(''.join(json.dumps(r)+'\n' for r in groups[axis] if str(r['video_id']) in changed))
                correction=RUN/('corrections/%s_%d'%(axis,side))
                call('correct_%s_%d'%(axis,side),['scripts/reproduce/r43_readers.py','--kind','spatial','--source-dir',ROOT,
                  '--recipe',RUN/'extractor_binding.json','--','--model',model,'--index',idx,'--video-dir',T/'video','--out-dir',correction,
                  '--max-side',str(side),'--batch','8' if side==672 and axis=='x' else '16','--no-generate'])
                feature_check(correction,changed,False,lambda:None)
            else:correction=RUN/('corrections/%s_%d'%(axis,side))
            native=ROOT/('outputs/spf_ytest_%d'%side if axis=='y' else 'outputs/spf_restore_test_%d'%side)
            features=merged_features(native,correction,RUN/('features/%s_%d'%(axis,side)),axis_ids,changed)
            output=RUN/('reference_%s_%d.jsonl'%(axis,side))
            if axis=='y':
                reference=ROOT/('outputs/experiments/sym42_yopt/final_%d'%side)
                call('reference_y_%d'%side,['tools/train_spatial_head.py','--predict','--model-dir',reference,'--feats',features,
                  '--index',RUN/'index_y.jsonl','--out',output,'--device','cpu'])
                call('candidate_y_%d'%side,['scripts/reproduce/r43_predict.py','--model-dir',OUT/('candidate_%d'%side),
                  '--feats',features,'--index',RUN/'index_y.jsonl','--out',RUN/('candidate_y_%d.jsonl'%side)])
            else:
                call('reference_x_%d'%side,['tools/train_spatial_head.py','--predict','--model-dir',ROOT/('outputs/sph_restore_%d_x'%side),
                  '--feats',features,'--index',RUN/'index_x.jsonl','--out',output,'--device','cpu'])
        key='center_'+axis;dest=RUN/('centers_'+axis+'.jsonl');average([RUN/('reference_%s_%d.jsonl'%(axis,s)) for s in [448,672]],key,dest);centers[axis]=dest
    candidate_y=RUN/'centers_candidate_y.jsonl';average([RUN/('candidate_y_%d.jsonl'%s) for s in [448,672]],'center_y',candidate_y)
    decode=json.loads((ROOT/'tools/configs/d3_decode.json').read_text())
    for shift in [0,3]:
        params=RUN/('decode%d.json'%shift);write(params,dict(decode,shift=float(shift)))
        call('decode%d'%shift,['tools/decode_scores.py','--probs',probs,'--params',params,'--index',T/'submission_index.jsonl',
             '--video-dir',T/'video','--safe-frames','--write-submission',RUN/('shift%d.jsonl'%shift)])
    import torch
    with shared.original_modules(ROOT) as (rf,tf,geometry):
        rf.iter_frames=reliable.detector_reader(rf.iter_frames)
        detector=rf.Detector('/root/sj-fs/autodl-tmp/sym/models/yolo11s.pt','cuda:0',640)
        detector_count=sum(p.numel() for p in detector.model.model.parameters());assert detector_count==9458752
        cfg=json.loads((ROOT/'tools/configs/final_AM.json').read_text());decoded=records(RUN/'shift0.jsonl');entries=[]
        for i,r in enumerate(groups['x']+groups['y']):
            v=str(r['video_id']);path=str(T/'video'/(v+'.mp4'));frames=[p['frame'] for p in decoded[v]['predictions']]
            detector.tag=rf.detector_tag('/root/sj-fs/autodl-tmp/sym/models/yolo11s.pt',640)+('+strict-capability-r43' if v in affected else '')
            cached=None if v in affected else rf.load_cached(path,frames,cfg['reframe']['stride'],detector.tag,'/root/sj-fs/work/repo/outputs/det_cache')
            if cached is None:cached=rf.collect(path,frames,cfg['reframe']['stride'],detector,str(RUN/'cache'))
            dst=RUN/'cache'/(v+'.json');dst.write_text(json.dumps(cached))
            entries.append({'video_id':v,'path':str(dst.relative_to(RUN)),'bytes':dst.stat().st_size,'sha256':digest(dst),'detector':cached['detector']})
            if i%50==0:print('DETECTION',i+1,254,flush=True)
    del detector;gc.collect();torch.cuda.empty_cache()
    def head_entries(paths):
        entries=[]
        for path in paths:
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert all(torch.isfinite(t).all() for t in state.values())
            entries.append(dict(path=str(path),parameters=sum(t.numel() for t in state.values()),sha256=digest(path)))
        return entries
    common=head_entries([ROOT/('outputs/vh_restore_baseline_L%d/fold%d.pt'%(l,f)) for l in [1,2] for f in range(5)]
                        +[ROOT/('outputs/sph_restore_%d_x/head.pt'%s) for s in [448,672]])
    reference_heads=head_entries([ROOT/('outputs/experiments/sym42_yopt/final_%d/head.pt'%s) for s in [448,672]])
    candidate_heads=head_entries([OUT/('candidate_%d/head.pt'%s) for s in [448,672]])
    backbone_count=json.loads((RUN/'model_view_ready.json').read_text())['backbone_parameters']
    base_parameters=backbone_count+detector_count+sum(e['parameters'] for e in common+reference_heads)
    trained_parameters=backbone_count+detector_count+sum(e['parameters'] for e in common+candidate_heads)
    assert base_parameters==shared.PARAMETERS and trained_parameters<=9_000_000_000
    write(RUN/'actual_parameter_counts.json',dict(backbone=backbone_count,detector=detector_count,common_heads=common,
          reference_y_heads=reference_heads,candidate_y_heads=candidate_heads,reference_total=base_parameters,trained_total=trained_parameters))
    variants=[];scaled_boxes={scale:{} for scale in [.93,.92]}
    with shared.original_modules(ROOT) as (rf,tf,geometry):
        for row in groups['full']:
            v=str(row['video_id']);m=metadata[v];W,H=m['width'],m['height'];tw,th=row['targetRatioWH']
            x,y,w=rf.place_boxes([0],W,H,row['targetRatioWH'])[0]
            for scale in scaled_boxes:
                w2=int(w*scale);h=w*th/tw;h2=w2*th/tw
                scaled_boxes[scale][v]=[min(max(0,int(round(x+(w-w2)/2))),W-w2),min(max(0,int(round(y+(h-h2)/2))),int(H-h2)),w2]
    for name,cy,params in [('reference',centers['y'],base_parameters),('trained',candidate_y,trained_parameters)]:
        cache={'shift0':RUN/'shift0.jsonl','shift3':RUN/'shift3.jsonl','centers_x':centers['x'],'centers_y':cy,'probabilities':probs}
        recipe={'sources':sources,'geometry_counts':{k:len(v) for k,v in groups.items()},'detection_cache':entries,
          'cache_inputs':{k:{'path':str(f.relative_to(RUN)),'bytes':f.stat().st_size,'sha256':digest(f)} for k,f in cache.items()}}
        dest=RUN/(name+'_s95');dest.mkdir();shared.PARAMETERS=params
        result=shared.render_fixed_method(type('Args',(),dict(cache_root=RUN,source_dir=ROOT,out_dir=dest))(),recipe,lambda:None)
        ref=dest/'submission.jsonl';variants.append(dict(name=name+'_s95',**package(ref,rows)))
        write(dest/'method_inputs.json',recipe)
        for scale in [.93,.92]:
            target=RUN/(name+'_s%d'%round(scale*100));target.mkdir();out=target/'submission.jsonl'
            original=records(ref)
            with out.open('x') as f:
                for row in original.values():
                    v=str(row['video_id'])
                    if v in scaled_boxes[scale]:row=dict(row,predictions=[dict(p,bboxes=scaled_boxes[scale][v]) for p in row['predictions']])
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
            changed=records(out)
            for v,r in changed.items():
                assert [p['frame'] for p in r['predictions']]==[p['frame'] for p in original[v]['predictions']]
                if v not in scaled_boxes[scale]:assert r==original[v]
            variants.append(dict(name=target.name,**package(out,rows)))
    write(RUN/'complete.json',{'complete':True,'variants':variants,'training_selection':json.loads((OUT/'selection.json').read_text()),
          'cotsheep_reference_payload_sha256':'046962d9518a085c9c00e7832635d6a8bb39ec6d8af239b0b45e1da9af9dd0b9',
          'reference_exact_match':variants[0]['sha256']=='046962d9518a085c9c00e7832635d6a8bb39ec6d8af239b0b45e1da9af9dd0b9',
          'experiment_source_sha256':{f.name:digest(f) for f in (ROOT/'scripts/reproduce').glob('r43*.py')},
          'official_score':None,'submitted':False})
    print('HYBRID_COMPLETE',[(v['name'],v['frames']) for v in variants],flush=True)


if __name__=='__main__':main()
