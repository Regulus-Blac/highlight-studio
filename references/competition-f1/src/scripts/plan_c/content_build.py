#!/usr/bin/env python3
"""Full-frame box rule (10-21): for the videos whose source ratio equals the target (box = whole frame), every box
becomes SCALE of the largest target-ratio box that fits the region, centered on the region; region = the whole frame,
or with CONTENT=1 the content region of outputs/content_box.jsonl (content_box.py) for videos with symmetric flat bars
(both side bars >= 5 % of W and within 3 % of W of each other, or the same for top / bottom; bars near black). Frames and all other
rows untouched. SCALE is one global hyperparameter shared by every full-frame video (final: 0.92).
usage: [CONTENT=1] content_build.py BASE NAME SCALE   (BASE = package name under outputs/sub)"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, 'tools')
sys.path.insert(0, 'scripts/plan_c')
import reframe as rf
from highlight_metric import derive_h
from motion_build import T, digest

base, name, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
out = Path('outputs/sub/%s' % name)
assert not out.with_suffix('.zip').exists(), 'package exists, not overwriting'
C = {}
if os.environ.get('CONTENT'):
    for r in map(json.loads, open('outputs/content_box.jsonl')):
        x0, y0, x1, y1 = r['box']
        W, H = r['W'], r['H']
        side = min(x0, W - x1) >= 0.05 * W and abs(x0 - (W - x1)) <= 0.03 * W
        tb = min(y0, H - y1) >= 0.05 * H and abs(y0 - (H - y1)) <= 0.03 * H
        if r['bars'] == 'flat' and (side or tb):
            cap = cv2.VideoCapture(str(T / 'video' / (r['video_id'] + '.mp4')))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
            g = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            cap.release()
            bars = [g[:, :x0], g[:, x1:]] if side else [g[:y0], g[y1:]]
            if max(b.mean() for b in bars) < 25:   # black bars, not a dark sky or ground (sunsets)
                C[r['video_id']] = r['box']
changed = []
with open(out.with_suffix('.jsonl'), 'w') as f:
    for r in map(json.loads, open('outputs/sub/%s.jsonl' % base)):
        v = r['video_id']
        cap = cv2.VideoCapture(str(T / 'video' / (v + '.mp4')))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if r['predictions'] and rf.geometry(W, H, r['targetRatioWH']) is None and (not C or v in C):
            tw, th = r['targetRatioWH']
            x0, y0, x1, y1 = C.get(v, [0, 0, W, H])
            w = int(min(x1 - x0, (y1 - y0) * tw / th) * scale)
            h = derive_h(w, [tw, th])
            x = min(max(0, int(round((x0 + x1 - w) / 2))), W - w)
            y = min(max(0, int(round((y0 + y1 - h) / 2))), int(H - h))
            new = dict(r, predictions=[dict(p, bboxes=[x, y, w]) for p in r['predictions']])
            if new != r:
                changed.append(v)
            r = new
        f.write(json.dumps(r) + '\n')
print('%s: full-frame boxes of %d videos at %.3f of %s' % (name, len(changed), scale, 'the content region' if C else 'the frame'),
      changed if C else '')
cmd = [sys.executable, 'tools/validate_submission.py', '--pred', str(out.with_suffix('.jsonl')),
       '--index', str(T / 'submission_index.jsonl'), '--video-dir', str(T / 'video')]
chk = subprocess.run(cmd, text=True, capture_output=True)
out.with_suffix('.validate.log').write_text(chk.stdout + chk.stderr)
chk.check_returncode()
with zipfile.ZipFile(out.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(out.with_suffix('.jsonl'), arcname='predictions.jsonl')
m = json.load(open('outputs/sub/%s.manifest.json' % base))
manifest = {'name': name, 'base': base, 'scale': scale, 'content': bool(C), 'videos_changed': len(changed),
            'parameters': m['parameters'], 'total_parameters': m['total_parameters'],
            'sha256': digest(out.with_suffix('.zip')), 'validation': 'PASS', 'submitted': False}
out.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('CONTENT BUILD DONE', out, manifest['sha256'][:16])
