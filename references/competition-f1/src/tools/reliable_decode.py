#!/usr/bin/env python3
"""Capability-selected decoder for the observed unsupported color-transfer failure.

Usage: python tools/reliable_decode.py infer_yesno [usual arguments]
No video-ID routing; supported streams retain the original reader.
"""
import ctypes
import importlib
import sys
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np


@lru_cache(None)
def metadata(path):
    site=Path(cv2.__file__).resolve().parents[1]
    library=next(site.glob('opencv_python_headless.libs/libswscale*.so*'))
    lib=ctypes.CDLL(str(library));supported=lib.sws_test_transfer
    supported.argtypes=[ctypes.c_int,ctypes.c_int];supported.restype=ctypes.c_int
    with av.open(path) as container:transfer=int(container.streams.video[0].codec_context.color_trc)
    cap=cv2.VideoCapture(path)
    n,fps,w,h=int(cap.get(7)),cap.get(5),int(cap.get(3)),int(cap.get(4));cap.release()
    return dict(supported=supported(transfer,0)>0,color_trc=transfer,n=n,fps=fps,w=w,h=h)


def fallback(path,wanted):
    import decord
    m=metadata(path);reader=decord.VideoReader(path,ctx=decord.cpu(0),num_threads=1)
    assert len(reader)==m['n'] and abs(reader.get_avg_fps()-m['fps'])<1e-6
    wanted=set(wanted)
    if not wanted:return
    for i in range(max(wanted)+1):
        frame=reader.next()
        if i in wanted:
            rgb=frame.asnumpy()
            assert rgb.shape==(m['h'],m['w'],3)
            yield i,rgb


def install():
    import infer_yesno as temporal
    import api_teacher as spatial
    import reframe
    import decode_scores
    old_t,old_s,old_r,old_n=temporal.read_frames,spatial.frames_1fps,reframe.iter_frames,decode_scores.decoded_frames
    old_detector_init=reframe.Detector.__init__

    def detector_init(self,*args,**kwargs):
        old_detector_init(self,*args,**kwargs)
        self.tag += '+capability-sequential-v1'

    def read_temporal(path,frame_ids,max_side,portrait_crop=False):
        if metadata(path)['supported']:return old_t(path,frame_ids,max_side,portrait_crop)
        result={}
        for i,rgb in fallback(path,frame_ids):
            h,w=rgb.shape[:2]
            if portrait_crop and w>h:
                cw=int(h*9/16);rgb=rgb[:,(w-cw)//2:(w-cw)//2+cw];h,w=rgb.shape[:2]
                scale=max_side/max(w,h)
                rgb=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_CUBIC)
            else:
                scale=min(1.,max_side/max(w,h))
                if scale<1:rgb=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
            result[i]=rgb
        return result

    def read_spatial(path,max_side):
        if metadata(path)['supported']:return old_s(path,max_side)
        m=metadata(path);n=int(np.floor((m['n']-1)/m['fps']))+1
        times={int(round(s*m['fps'])):s for s in range(n)};result=[]
        for i,rgb in fallback(path,times):
            h,w=rgb.shape[:2];scale=min(1.,max_side/max(w,h))
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            if scale<1:bgr=cv2.resize(bgr,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
            ok,jpg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok
            result.append((times[i],jpg.tobytes()))
        return result

    def read_detector(path,wanted):
        if metadata(path)['supported']:yield from old_r(path,wanted)
        else:
            for i,rgb in fallback(path,wanted):yield i,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

    def count_frames(path):
        if metadata(path)['supported']:return old_n(path)
        return sum(1 for _ in fallback(path,range(metadata(path)['n'])))

    temporal.read_frames=read_temporal;spatial.frames_1fps=read_spatial
    reframe.iter_frames=read_detector;decode_scores.decoded_frames=count_frames
    reframe.Detector.__init__=detector_init


if __name__=='__main__':
    entry=sys.argv.pop(1)
    assert entry in ('infer_yesno','spatial_feats','decode_scores')
    install()
    sys.exit(importlib.import_module(entry).main())
