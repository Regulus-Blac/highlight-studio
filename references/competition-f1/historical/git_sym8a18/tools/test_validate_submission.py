#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the submission format gate / 提交格式校验器的测试.

    python3 tools/test_validate_submission.py

Builds a clean submission that must PASS, then injects one fault class at a
time and asserts the validator FAILs and names it. Runs the real CLI through
a subprocess so the exit code is covered too.
逐类注入错误，确认校验器报错且指出原因；通过子进程跑真实 CLI，顺带覆盖退出码。

Temp files go to a scratch dir (--workdir), never into the repo.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATOR = os.path.join(HERE, "validate_submission.py")

# A small stand-in index: two 16:9 videos and one 9:16, all 534x300 @ 100 frames.
INDEX = [
    {"video_id": "a", "targetRatioWH": [16, 9]},
    {"video_id": "b", "targetRatioWH": [16, 9]},
    {"video_id": "c", "targetRatioWH": [9, 16]},
]
META = {v["video_id"]: {"n_frames": 100, "width": 534, "height": 300} for v in INDEX}

PASSED, FAILED = [], []


def clean_submission():
    """A submission that satisfies every hard constraint."""
    recs = []
    for it in INDEX:
        vid, (tw, th) = it["video_id"], it["targetRatioWH"]
        W, H = 534, 300
        # floor(), not round(): round() can push the derived h past the frame.
        if W / float(H) >= tw / float(th):
            cw = int(W * 0 + (H * tw) // th)
        else:
            cw = W
        cw = min(cw, W)
        ch = cw * th // tw
        x, y = (W - cw) // 2, (H - ch) // 2
        preds = [{"frame": f, "bboxes": [x, y, cw]} for f in range(10, 40)]
        recs.append({"video_id": vid, "targetRatioWH": [tw, th], "predictions": preds})
    return recs


def run_validator(recs, workdir, extra=(), jsonl_index=False):
    pred_path = os.path.join(workdir, "pred.jsonl")
    index_path = os.path.join(workdir, "index.jsonl" if jsonl_index else "index.json")
    meta_path = os.path.join(workdir, "meta.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(index_path, "w", encoding="utf-8") as f:
        if jsonl_index:  # the 复赛 index ships as JSONL, one object per line
            for it in INDEX:
                f.write(json.dumps(it) + "\n")
        else:
            json.dump(INDEX, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(META, f)
    cmd = [sys.executable, VALIDATOR, "--pred", pred_path,
           "--index", index_path, "--meta", meta_path] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def expect(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  [%s] %s%s" % ("ok" if cond else "XX", name,
                           "" if cond else "   <- " + detail))


def case(name, mutate, workdir, needle):
    """Apply `mutate` to a clean submission; expect FAIL mentioning `needle`."""
    recs = clean_submission()
    mutate(recs)
    rc, out = run_validator(recs, workdir)
    hit = needle.lower() in out.lower()
    expect(name, rc != 0 and hit,
           "rc=%d needle=%r not found" % (rc, needle) if not hit else "rc=%d" % rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for temp files (default: system temp)")
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="subval_")
    made = not args.workdir
    os.makedirs(workdir, exist_ok=True)
    print("  workdir: %s" % workdir)
    print("-" * 78)

    try:
        # ---- the happy path must pass -------------------------------
        rc, out = run_validator(clean_submission(), workdir)
        expect("clean submission passes", rc == 0 and "PASS" in out, out[-400:])
        rc, out = run_validator(clean_submission(), workdir, jsonl_index=True)
        expect("clean submission passes with a JSONL index", rc == 0 and "PASS" in out, out[-400:])

        # ---- one fault class at a time ------------------------------
        print("-" * 78)

        def drop_video(r):
            r.pop()
        case("missing video_id from the index", drop_video, workdir,
             "has no line in the submission")

        def unknown_video(r):
            r.append({"video_id": "zzz", "targetRatioWH": [16, 9], "predictions": []})
        case("video_id not in the index", unknown_video, workdir,
             "is not in the index")

        def dup_video(r):
            r.append(dict(r[0]))
        case("duplicate video_id line", dup_video, workdir, "duplicate video_id")

        def altered_ratio(r):
            r[0]["targetRatioWH"] = [9, 16]
        case("targetRatioWH altered from the index", altered_ratio, workdir,
             "differs from the index value")

        def quad_bbox(r):
            r[0]["predictions"][0]["bboxes"] = [0, 0, 100, 56]
        case("bboxes submitted as a quadruple", quad_bbox, workdir,
             "must be a triplet")

        def nan_bbox(r):
            r[0]["predictions"][0]["bboxes"] = [0, 0, float("nan")]
        case("bboxes contains NaN", nan_bbox, workdir, "nan")

        def zero_w(r):
            r[0]["predictions"][0]["bboxes"][2] = 0
        case("w <= 0", zero_w, workdir, "must be > 0")

        def neg_x(r):
            r[0]["predictions"][0]["bboxes"][0] = -1
        case("x < 0", neg_x, workdir, "must be >= 0")

        def frame_oob(r):
            r[0]["predictions"][0]["frame"] = 100000
        case("frame beyond n_frames", frame_oob, workdir, "n_frames")

        def frame_neg(r):
            r[0]["predictions"][0]["frame"] = -5
        case("frame < 0", frame_neg, workdir, "frame -5 < 0")

        def dup_frame(r):
            r[0]["predictions"].append(dict(r[0]["predictions"][0]))
        case("duplicate frame in one video", dup_frame, workdir, "duplicate frame")

        def wide_box(r):
            r[0]["predictions"][0]["bboxes"] = [500, 0, 500]
        case("x+w past the frame width", wide_box, workdir, "exceeds width")

        def tall_box(r):
            # the round()-vs-floor() trap on the 9:16 video: w=169 -> h=300.44
            r[2]["predictions"][0]["bboxes"] = [182, 0, 169]
        case("derived h past the frame height (round/floor trap)", tall_box,
             workdir, "exceeds height")

        def missing_field(r):
            r[0]["predictions"][0] = {"frame": 10}
        case("prediction missing bboxes", missing_field, workdir, "missing field")

        recs = clean_submission()
        pred_path = os.path.join(workdir, "pred.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(recs[0]) + "\n")
            f.write("{not json at all\n")
        index_path = os.path.join(workdir, "index.json")
        meta_path = os.path.join(workdir, "meta.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(INDEX, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(META, f)
        pr = subprocess.run([sys.executable, VALIDATOR, "--pred", pred_path,
                             "--index", index_path, "--meta", meta_path],
                            capture_output=True, text=True)
        expect("malformed JSONL line", pr.returncode != 0
               and "invalid json" in (pr.stdout or "").lower(), pr.stdout[-300:])

        # ---- missing metadata is an error unless explicitly allowed --
        print("-" * 78)
        recs = clean_submission()
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        pr = subprocess.run([sys.executable, VALIDATOR, "--pred", pred_path,
                             "--index", index_path], capture_output=True, text=True)
        expect("no metadata -> fails by default", pr.returncode != 0
               and "could not run" in (pr.stdout or "").lower(), pr.stdout[-300:])
        pr = subprocess.run([sys.executable, VALIDATOR, "--pred", pred_path,
                             "--index", index_path, "--allow-unchecked"],
                            capture_output=True, text=True)
        expect("no metadata + --allow-unchecked -> passes with a warning",
               pr.returncode == 0 and "[W]" in (pr.stdout or ""), pr.stdout[-300:])

        # ---- warnings must not fail the run --------------------------
        recs = clean_submission()
        recs[0]["predictions"] = []
        rc, out = run_validator(recs, workdir)
        expect("empty predictions is a warning, not an error",
               rc == 0 and "empty predictions" in out, out[-300:])

        recs = clean_submission()
        recs[0]["predictions"].reverse()
        rc, out = run_validator(recs, workdir)
        expect("descending frames is a warning, not an error",
               rc == 0 and "ascending order" in out, out[-300:])

        recs = clean_submission()
        x, y, cw = (recs[0]["predictions"][0]["bboxes"])
        recs[0]["predictions"] = [{"frame": f, "bboxes": [x, y, cw]}
                                  for f in range(100)]
        rc, out = run_validator(recs, workdir)
        expect("near-full frame coverage is a warning",
               rc == 0 and "near-full coverage" in out, out[-300:])

    finally:
        if made:
            shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 78)
    print("  %d passed, %d failed" % (len(PASSED), len(FAILED)))
    for n in FAILED:
        print("    FAILED: %s" % n)
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
