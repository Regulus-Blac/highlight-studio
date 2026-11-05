#!/usr/bin/env python3
"""Replace the boxes of a submission's free-axis rows with crops placed at given per-second subject centers
(median over +-1 s), frames untouched; other rows copied. Centers come from local Qwen3-VL-8B grounding or the
learned spatial head (JSONL with center_x and / or center_y per second). Local automated inference only.
usage: ground_build.py BASE_SUB.jsonl CENTERS_X.jsonl OUT.jsonl [CENTERS_Y.jsonl]"""
import json, os, sys
import numpy as np
import cv2
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import reframe as rf
from highlight.geometry import place_1d, safe_crop_size, dof_range
base, gx, out = sys.argv[1:4]
gy = sys.argv[4] if len(sys.argv) > 4 else None
T = os.environ.get("TEST_VIDEO_DIR", "/root/sj-fs/autodl-tmp/public/aic-competition-test/2026_semifinal/video")
def load(path, key):
    if not path:
        return {}
    return {str(r["video_id"]): np.asarray(r[key], float) for r in map(json.loads, open(path)) if "error" not in r and key in r}
cen = {"x": load(gx, "center_x"), "y": load(gy, "center_y")}
def medfilt(x, w=1):
    return np.array([np.median(x[max(0, i - w):i + w + 1]) for i in range(len(x))])
n = {"x": 0, "y": 0}
with open(out, "w") as f:
    for line in open(base):
        row = json.loads(line)
        vid = str(row["video_id"])
        if row.get("predictions"):
            path = "%s/%s.mp4" % (T, vid)
            W, H = rf.video_size(path)
            tw, th = [float(v) for v in row["targetRatioWH"]]
            geo = rf.geometry(W, H, [tw, th])
            if geo is not None and vid in cen[geo[0]]:
                axis = geo[0]
                cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; cap.release()
                c = medfilt(cen[axis][vid]); w, h = safe_crop_size(W, H, tw, th); R = dof_range(W, H, tw, th)
                L, ext = (W, w) if axis == "x" else (H, h)
                preds = []
                for p in row["predictions"]:
                    cc = float(np.interp(int(p["frame"]) / fps, np.arange(len(c)), c))
                    u = min(1.0, max(0.0, (cc * L - ext / 2.0) / R)) if R > 0 else 0.5
                    x0, y0, w0 = place_1d(u, W, H, tw, th)
                    preds.append(dict(p, bboxes=[int(x0), int(y0), int(w0)]))
                row = dict(row, predictions=preds); n[axis] += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("re-boxed rows: x %d, y %d -> %s" % (n["x"], n["y"], out))
