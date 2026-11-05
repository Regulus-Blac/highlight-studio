#!/usr/bin/env python3
"""Content region of each test video (10-21): 32 of the 172 full-frame videos carry letterbox / pillarbox bars (mean
24 % of H / 34 % of W), so a box over the whole frame (or 93 % of it) also covers the bars. Bars are the edge bands of
flat rows / columns (sd < 10 on all 8 sampled frames: black, white or colored) plus blurred pads (edge bands whose
Laplacian energy stays below 15 % of the middle third's on all frames). Only bands touching an edge count.
Writes outputs/content_box.jsonl {video_id, W, H, box: [x0, y0, x1, y1] (end exclusive), bars: flat / blur / none}.
usage: content_box.py   (TEST_VIDEO_DIR as ground_build.py)"""
import json
import os

import cv2
import numpy as np

T = os.environ.get('TEST_VIDEO_DIR', '/Volumes/SSD/aic/semi_test/video')


def band(flat):
    """edge bands of a boolean profile -> (start, end) of the middle part"""
    a, b = 0, len(flat)
    while a < b and flat[a]:
        a += 1
    while b > a and flat[b - 1]:
        b -= 1
    return a, b


ids = [r['video_id'] for r in map(json.loads, open('/Volumes/SSD/aic/semi_test/submission_index.jsonl'))]
n_bars = 0
with open('outputs/content_box.jsonl', 'w') as out:
    for v in ids:
        cap = cv2.VideoCapture('%s/%s.mp4' % (T, v))
        n, W, H = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        flat_r, flat_c, blur_r, blur_c = [], [], [], []
        for f in np.linspace(0.05 * n, 0.95 * n, 8).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
            flat_r.append(g.std(1) < 10)
            flat_c.append(g.std(0) < 10)
            er, ec = lap.mean(1), lap.mean(0)
            blur_r.append(er < 0.15 * np.median(er[H // 3:2 * H // 3]))
            blur_c.append(ec < 0.15 * np.median(ec[W // 3:2 * W // 3]))
        cap.release()
        y0, y1 = band(np.all(flat_r, 0))
        x0, x1 = band(np.all(flat_c, 0))
        kind = 'flat' if (y1 - y0) * (x1 - x0) < 0.95 * W * H else 'none'
        if kind == 'none':
            by0, by1 = band(np.all(blur_r, 0))
            bx0, bx1 = band(np.all(blur_c, 0))
            if (by1 - by0) * (bx1 - bx0) < 0.95 * W * H:
                y0, y1, x0, x1, kind = by0, by1, bx0, bx1, 'blur'
        if kind == 'none':
            x0, y0, x1, y1 = 0, 0, W, H
        n_bars += kind != 'none'
        out.write(json.dumps({'video_id': v, 'W': W, 'H': H, 'box': [int(x0), int(y0), int(x1), int(y1)], 'bars': kind}) + '\n')
print('%d of %d videos have bars -> outputs/content_box.jsonl' % (n_bars, len(ids)))
