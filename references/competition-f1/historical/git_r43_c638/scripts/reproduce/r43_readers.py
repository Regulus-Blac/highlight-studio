"""Use cotsheep's capability routing with strict sequential Decord and RGB/PTS audits."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/root/sj-fs/work/sym41_8a18db2')
OUT=ROOT/'outputs/experiments/sym43_crop_context'
COT=Path('/root/sj-fs/work/cotsheep_a4809a1')
sys.path.insert(0,str(COT))
from aic.data import reliable_video as reliable


def strict_rgb(path,indices,metadata):
    import decord
    wanted=reliable._indices(indices,metadata['frames'])
    if not wanted:return {}
    reader=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=1,fault_tol=1e-12)
    assert len(reader)==metadata['frames']
    assert math.isclose(reader.get_avg_fps(),metadata['fps'],rel_tol=1e-9,abs_tol=1e-9)
    selected=set(wanted);result={};digest=hashlib.sha256()
    for index in range(max(wanted)+1):
        frame=reader.next().asnumpy()
        assert frame.dtype==np.uint8 and frame.shape==(metadata['height'],metadata['width'],3)
        digest.update(frame.tobytes())
        if index in selected:result[index]=frame.copy()
    assert set(result)==selected
    entry={'path':str(path),'requested':wanted,'decoded_rgb_sha256':digest.hexdigest(),'decoded_frames':max(wanted)+1,
           'frame_rgb_sha256':{str(i):hashlib.sha256(v.tobytes()).hexdigest() for i,v in result.items()},
           'decoder':'decord_cpu_sequential','fault_tol':1e-12}
    with (OUT/'decoder_reads.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    return result


reliable.fallback_rgb=strict_rgb


def qualify():
    import av,decord
    metadata=json.loads((OUT/'decoder_metadata.json').read_text())
    video_dir=Path('/root/sj-fs/autodl-tmp/public/aic-competition-test/2026_semifinal/video')
    results={}
    for vid,m in metadata.items():
        if m['opencv_transfer_supported']:continue
        path=video_dir/(vid+'.mp4')
        ids=sorted({min(m['frames']-1,int(round(s*m['fps']))) for s in range(int(np.floor((m['frames']-1)/m['fps']))+1)}|{m['frames']-1})
        hashes=[]
        for _ in range(3):
            frames=strict_rgb(path,ids,m)
            hashes.append({str(i):hashlib.sha256(v.tobytes()).hexdigest() for i,v in frames.items()})
            del frames
        assert hashes[0]==hashes[1]==hashes[2]
        with av.open(str(path)) as container:
            pts=np.array([float(f.pts*f.time_base) for f in container.decode(video=0)])
        reader=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=1,fault_tol=1e-12)
        dpts=reader.get_frame_timestamp(np.arange(len(reader)))[:,0]
        assert len(pts)==len(dpts)==m['frames'] and (np.diff(pts)>0).all()
        error=float(np.max(np.abs(pts-dpts)))
        assert error<1e-5
        results[vid]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'repeat_rgb_identical':True,'pts_max_error_seconds':error,'frames':len(pts)}
    (OUT/'decoder_qualification.json').write_text(json.dumps({'complete':True,'sources':results},indent=2))
    print('DECODER_QUALIFIED',results,flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--qualify']:qualify()
    else:
        from aic.inference.checked_shared_script import main
        main()
