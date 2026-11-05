#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download only the source videos train.jsonl points at / 只下载标签用到的源片.

The QVHighlights mirror on Hugging Face (ayushsdev/qvhighlights-videos) holds
12.5k videos, 145 GB, laid out as <first char lowercased>/<source_vid>.mp4 --
the same layout as the Baidu share. train.jsonl uses 987 of them (~11 GB), so
this lists the repo once, keeps the files whose name is a
`clip.source_vid`, and downloads just those. Files already under --out-dir
(any subfolder, e.g. a partial Baidu copy) are skipped, so it doubles as a
"fill the gaps" tool.
HF 镜像 145 GB，训练只用 987 个（约 11 GB）；已有的跳过，可用于补缺。

    # server (mainland: add HF_ENDPOINT=https://hf-mirror.com)
    python3 tools/fetch_sources.py --train train.jsonl --out-dir data/video
    # just print what is missing
    python3 tools/fetch_sources.py --train train.jsonl --out-dir /Volumes/Untitled/训练集 --dry-run

Then: build_train_clips.py --source-dir data/video (it walks subfolders).
"""

import os
import sys
import json
import time
import argparse

REPO = "ayushsdev/qvhighlights-videos"


def needed_sources(train_path):
    need = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                need.add(json.loads(line)["clip"]["source_vid"])
    return need


def local_sources(root):
    have = set()
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".mp4") and not fn.startswith("._"):
                have.add(fn[:-4])
    return have


def main():
    ap = argparse.ArgumentParser(description="Fetch the QVHighlights source videos train.jsonl uses")
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--out-dir", default="data/video")
    ap.add_argument("--also-have", nargs="*", default=[],
                    help="extra dirs that already hold sources (skipped, not copied)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="list missing files only")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("[fatal] pip install huggingface_hub")

    need = needed_sources(args.train)
    have = local_sources(args.out_dir) if os.path.isdir(args.out_dir) else set()
    for d in args.also_have:
        have |= local_sources(d)
    miss = need - have
    print("sources in %s: %d   already local: %d   to fetch: %d"
          % (args.train, len(need), len(need & have), len(miss)))
    if not miss:
        return 0

    t0 = time.time()
    tree = [f for f in HfApi().list_repo_tree(args.repo, repo_type="dataset", recursive=True)
            if getattr(f, "size", None) is not None and f.path.endswith(".mp4")]
    by_name = {os.path.basename(f.path)[:-4]: f for f in tree}
    want = sorted(by_name[v].path for v in miss if v in by_name)
    absent = sorted(miss - set(by_name))
    size_gb = sum(by_name[v].size for v in miss if v in by_name) / 1e9
    print("repo files: %d   fetchable: %d (%.2f GB)   not in repo: %d   (listing %.0fs)"
          % (len(tree), len(want), size_gb, len(absent), time.time() - t0))
    for v in absent[:10]:
        print("  [missing upstream] %s" % v)
    if args.dry_run:
        for p in want:
            print(p)
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    snapshot_download(args.repo, repo_type="dataset", allow_patterns=want,
                      local_dir=args.out_dir, max_workers=args.workers)
    got = local_sources(args.out_dir) | set().union(*(local_sources(d) for d in args.also_have))
    left = need - got
    print("done in %.0fs   still missing: %d" % (time.time() - t0, len(left)))
    return 1 if left - set(absent) else 0


if __name__ == "__main__":
    sys.exit(main())
