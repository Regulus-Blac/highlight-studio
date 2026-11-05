#!/usr/bin/env python3
"""Portrait-strip labels from a strong hosted VLM (user 10-17: a 45-point team trained its 16:9 composition heads on
boxes from other models). TRAINING videos only (data/yhead_train.jsonl, deduplicated against the test set). The model
boxes the main subject of every 1-fps frame (0-1000 coordinates) and names its type; the strip center then follows
our leaderboard-validated anchoring (R1T_crop3: 0.20 down a person / object box, 0.35 down an animal box), clamped so
the 16:9 strip stays in the frame and median-filtered over +-1 s. Seconds without a subject are interpolated.
--head (user 10-17): the model also boxes the subject's head (face included); close-ups then aim at the head -- the
strip center moves from the anchor point to the head center as the head grows from 10 % to 20 % of the frame height
(close-up = big head; a full-body person keeps the anchor: face-point strips lost on the preliminary leaderboard,
41.91 vs 42.45), and a head whose center still falls outside the strip is brought just inside it (pets walking at the
camera, heads raised). The heads trained on anchor-only labels placed close-up pets too high.
usage: ybox_labels.py MODEL OUT_ANSWERS.jsonl OUT_TRAIN.jsonl [--rows ROWS.jsonl] [--videos V.jsonl ...] [--limit N]
       [--budget CNY] [--head]   (rows default data/yhead_train.jsonl; videos give the frame sizes)"""
import argparse
import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, 'tools')
from api_teacher import frames_1fps

PRICES = {'qwen/qwen3.8-max': (12.0, 36.0), 'qwen/qwen3.8-flash': (0.8, 2.7), 'qwen/qwen3.7-plus': (2.0, 8.0)}
PROMPT = ("上面是同一段竖屏短视频按每秒 1 帧抽取的 {n} 张画面，按时间顺序排列，每张前面标了它对应的秒数。\n"
          "逐秒框出这一秒画面里最值得观众看到的主体（通常是视频在拍的人或宠物；拍摄者从画面边缘伸进来的手、腿不算主体），"
          "给出它的完整可见范围 bbox = [x1, y1, x2, y2]（0 到 1000 的整数坐标，相对这一帧，左上到右下），"
          "以及类别 type（person、animal 或 other）。这一秒没有明确主体时 bbox 给 null。\n"
          '只输出 JSON：{{"timeline": [{{"time_sec": 0, "bbox": [x1, y1, x2, y2], "type": "person"}}, ...]}}，'
          "按顺序覆盖第 0 到第 {last} 秒，共 {n} 项。")
HEAD_PROMPT = ("上面是同一段竖屏短视频按每秒 1 帧抽取的 {n} 张画面，按时间顺序排列，每张前面标了它对应的秒数。\n"
               "逐秒框出这一秒画面里最值得观众看到的主体（通常是视频在拍的人或宠物；拍摄者从画面边缘伸进来的手、腿不算主体），"
               "给出它的完整可见范围 bbox = [x1, y1, x2, y2]（0 到 1000 的整数坐标，相对这一帧，左上到右下）、"
               "类别 type（person、animal 或 other），以及这个主体的头部范围 head = [x1, y1, x2, y2]"
               "（人和动物都框整个头，包括脸；背对镜头时框后脑；看不到头或主体是物体时 head 给 null）。"
               "这一秒没有明确主体时 bbox 和 head 都给 null。\n"
               '只输出 JSON：{{"timeline": [{{"time_sec": 0, "bbox": [x1, y1, x2, y2], "type": "person", '
               '"head": [x1, y1, x2, y2]}}, ...]}}，按顺序覆盖第 0 到第 {last} 秒，共 {n} 项。')

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('answers')
ap.add_argument('train')
ap.add_argument('--rows', default='data/yhead_train.jsonl')
ap.add_argument('--videos', nargs='+', default=['data/pilot_videos.jsonl', 'data/pl2_videos.jsonl'])
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--budget', type=float, default=40.0)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--head', action='store_true')
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.rows)][:args.limit or None]
assert not any('test' in r['video_path'].lower() or 'semifinal' in r['video_path'].lower() for r in rows)
key = open('/root/sj-fs/.ppio_key').read().strip()
p_in, p_out = PRICES[args.model]
lock, st = threading.Lock(), {'cost': 0.0, 'ok': 0, 'err': 0}


def work(r):
    with lock:
        if st['cost'] >= args.budget:
            return
    rec, t0 = {'video_id': r['video_id'], 'model': args.model}, time.time()
    try:
        frames = frames_1fps(r['video_path'], 448)
        content = []
        for sec, jpg in frames:
            img = 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()
            content += [{'type': 'text', 'text': '第 %d 秒' % sec}, {'type': 'image_url', 'image_url': {'url': img}}]
        prompt = HEAD_PROMPT if args.head else PROMPT
        content.append({'type': 'text', 'text': prompt.format(n=len(frames), last=len(frames) - 1)})
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 6000,
                'temperature': 0.2, 'enable_thinking': False}
        resp = requests.post('https://api.ppinfra.com/v3/openai/chat/completions', json=body, timeout=600,
                             headers={'Authorization': 'Bearer ' + key})
        assert resp.status_code == 200, 'HTTP %d: %s' % (resp.status_code, resp.text[:300])
        d = resp.json()
        text, usage = d['choices'][0]['message'].get('content') or '', d.get('usage') or {}
        m = re.search(r'\{.*\}', re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M), flags=re.S)
        tl = json.loads(m.group(0))['timeline'] if m else []
        rec.update(n_sec=len(frames), usage=usage, latency=round(time.time() - t0, 1),
                   boxes={int(round(float(e['time_sec']))): [e.get('bbox'), e.get('type')] for e in tl})
        if args.head:
            rec['heads'] = {int(round(float(e['time_sec']))): e.get('head') for e in tl}
        if not tl:
            rec.update(error='unparsable reply', raw=text[:300])
    except Exception as e:  # one bad video must not stop the batch
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
    with lock:
        u = rec.get('usage') or {}
        st['cost'] += (u.get('prompt_tokens', 0) * p_in + u.get('completion_tokens', 0) * p_out) / 1e6
        st['err' if 'error' in rec else 'ok'] += 1
        with open(args.answers, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print('  %s %s in %s out %s | CNY %.3f' % (r['video_id'], rec.get('error', 'ok')[:60], u.get('prompt_tokens'),
                                                    u.get('completion_tokens'), st['cost']), flush=True)


done = set()
try:
    done = {d['video_id'] for d in map(json.loads, open(args.answers)) if 'error' not in d}
except FileNotFoundError:
    pass
with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(work, [r for r in rows if r['video_id'] not in done]))
print('labeled: %d ok, %d errors, CNY %.3f' % (st['ok'], st['err'], st['cost']), flush=True)

answers = {d['video_id']: d for d in map(json.loads, open(args.answers)) if 'error' not in d}
dims = {v['video_id']: (v['width'], v['height']) for f in args.videos for v in map(json.loads, open(f))}
n_rows, kinds = 0, {'person': 0, 'animal': 0, 'other': 0, 'none': 0, 'head': 0}
with open(args.train, 'w') as f:
    for r in rows:
        d = answers.get(r['video_id'])
        if d is None:
            continue
        W, H = dims[r['video_id']]
        half = 9 / 32 * W / H   # half the strip height (w = W, h = 9 / 16 W) over the frame height
        n = d['n_sec']
        cen = np.full(n, np.nan)
        for k in range(n):
            box, kind = d['boxes'].get(str(k), d['boxes'].get(k, [None, None]))
            if not box or len(box) != 4:
                kinds['none'] += 1
                continue
            kind = kind if kind in ('person', 'animal') else 'other'
            kinds[kind] += 1
            y1, y2 = sorted((box[1] / 1000.0, box[3] / 1000.0))
            cen[k] = y1 + (0.35 if kind == 'animal' else 0.20) * (y2 - y1)
            hb = d.get('heads', {}).get(str(k))
            if hb and len(hb) == 4:
                h1, h2 = sorted((hb[1] / 1000.0, hb[3] / 1000.0))
                w = np.clip((h2 - h1 - 0.1) / 0.1, 0, 1)
                c = w * (h1 + h2) / 2 + (1 - w) * cen[k]
                if abs((h1 + h2) / 2 - c) > half:
                    c = min(max(c, h2 - half), h1 + half)
                kinds['head'] += abs(c - cen[k]) > 0.01
                cen[k] = c
        if np.isnan(cen).all():
            continue
        ok = ~np.isnan(cen)
        cen = np.interp(np.arange(n), np.flatnonzero(ok), cen[ok])
        cen = np.clip(cen, half, 1 - half)
        cen = np.array([np.median(cen[max(0, k - 1):k + 2]) for k in range(n)])
        traj = [{'time_sec': k, 'center': [0.5, round(float(c), 4)]} for k, c in enumerate(cen)]
        f.write(json.dumps(dict(r, teacher_signals={'trajectory': traj}, label_source=args.model)) + '\n')
        n_rows += 1
print('training rows: %d -> %s; seconds by subject type %s' % (n_rows, args.train, kinds))
