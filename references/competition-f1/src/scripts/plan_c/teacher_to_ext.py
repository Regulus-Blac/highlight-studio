#!/usr/bin/env python3
"""API-teacher timelines on contest clips -> ext rows for train_vlm_head.py --ext.

Each row reuses the contest clip id, path and source_group (so the folds keep a held-out clip's teacher row out
of training, and the val clips' rows never train), has no segments, and carries the teacher's per-second scores
as teacher_signals.timeline -- the ext channel learns that curve; the contest channels still learn doubao.
    python scripts/plan_c/teacher_to_ext.py outputs/teacher/step5_train.jsonl step5 data/ext_step5_contest.jsonl
"""
import json
import sys

src, name, out = sys.argv[1:4]
GT = "data/train.with_video.jsonl"
rows = {json.loads(l)["video_id"]: json.loads(l) for l in open(GT, encoding="utf-8")}
n = 0
with open(out, "w", encoding="utf-8") as f:
    for line in open(src, encoding="utf-8"):
        t = json.loads(line)
        if "error" in t or t["video_id"] not in rows:
            continue
        r = rows[t["video_id"]]
        f.write(json.dumps({
            "video_id": r["video_id"], "video_path": r["video_path"], "targetRatioWH": r["targetRatioWH"],
            "dataset_split": "ext", "segments": [],
            "teacher_signals": {"timeline": [{"time_sec": float(k), "highlight_score": float(s), "confidence": 1.0}
                                             for k, s in enumerate(t["scores"])]},
            "provenance": {"source": name, "source_group": r["provenance"]["source_group"]}}, ensure_ascii=False) + "\n")
        n += 1
print("%d rows -> %s" % (n, out))
