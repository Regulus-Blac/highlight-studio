#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared model / input plumbing for temporal SFT training and inference.
时序 SFT 训练与推理共用的模型加载与输入构造。

Both `train_sft_qlora.py` and `infer_temporal.py` go through `build_inputs`,
so the frames, pixel budget, timestamps and prompt the model sees at inference
are byte-for-byte what it saw in training. 训练与推理走同一条输入通路。

Frames are decoded here with OpenCV rather than by qwen-vl-utils: on
torchvision >= 0.25 `torchvision.io.read_video` no longer exists and
qwen-vl-utils' default reader fails. We sample the frames ourselves and hand
the processor a `VideoMetadata` carrying the real source fps and the sampled
indices, so Qwen2.5-VL's absolute-time mRoPE (`second_per_grid_ts`) matches
the seconds used in the targets.
自己用 OpenCV 抽帧：新版 torchvision 已删 read_video。帧索引与源 fps 通过
VideoMetadata 交给处理器，保证 mRoPE 的时间轴与标签秒数一致。
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_temporal import (render_prompt, render_score_prompt,  # noqa: E402
                                score_steps)  # single source of the prompts

# 8 GB budget, see docs/ENV_SETUP.md: 16 frames x 128 tokens / temporal merge 2
# = 1024 video tokens. 帧数与像素上限来自显存预算表。
DEFAULT_NFRAMES = 16
DEFAULT_MAX_PIXELS = 128 * 28 * 28      # 100352
DEFAULT_MIN_PIXELS = 32 * 28 * 28

# Per-second scoring task: two frames per one-second step, i.e. exactly one
# temporal patch, so every step gets its own `<t seconds>` stamp. Qwen3.5 packs
# 32x32 px per token; ~64 tokens per step keeps a 32-step clip near 2k tokens.
# 逐秒任务：每步 2 帧 = 一个时间 patch；每步约 64 token，32 步约 2k token。
STEP_FPS = 2
SCORE_MAX_PIXELS = 64 * 32 * 32         # 65536
SCORE_MIN_PIXELS = 16 * 32 * 32

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                # Qwen3.5 Gated DeltaNet (linear-attention) layers, 3 of every 4.
                # Qwen3.5 线性注意力层的投影，Qwen2.5 里不存在，不受影响。
                "in_proj_qkv", "in_proj_z", "out_proj")
# LoRA only on the language model; the vision tower stays frozen.
# 只在 LLM 上挂 LoRA，视觉塔冻结。
LORA_TARGET_REGEX = r"^(?!.*visual).*\.(%s)$" % "|".join(LORA_TARGETS)


def augment_frames(frames, augment):
    """Training-time augmentation of a list of RGB frames (H, W, 3).

    augment = {"flip": bool, "portrait": None | {"cx_px": float | None}}
      portrait: crop a 9:16 column out of a landscape frame, centered on
                `cx_px` (the labeled subject, None = frame center). The test
                set is mostly portrait sources the 534x300 training clips never
                show; highlight timing does not depend on the crop.
      flip    : mirror left-right (after the crop).
    训练期增强：竖屏模拟（按主体位置切 9:16）+ 水平翻转，时序标签不变。
    """
    if not augment or not frames:
        return frames
    port = augment.get("portrait")
    if port is not None:
        h, w = frames[0].shape[:2]
        cw = int(h * 9 // 16)
        if 0 < cw < w:
            cx = port.get("cx_px")
            cx = w / 2.0 if cx is None else float(cx)
            x0 = int(round(min(max(cx - cw / 2.0, 0.0), w - cw)))
            frames = [np.ascontiguousarray(f[:, x0:x0 + cw]) for f in frames]
    if augment.get("flip"):
        frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
    return frames


def sample_video(path, nframes=DEFAULT_NFRAMES, augment=None):
    """Uniformly sample `nframes` RGB frames -> (uint8 tensor T,C,H,W, metadata).

    Indices follow the HF sampler (`arange(0, N, N/n)`), so frame i sits at
    i / sampled_fps seconds. 帧 i 对应 i / sampled_fps 秒。
    """
    from transformers.video_utils import VideoMetadata

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 2:
        cap.release()
        raise IOError("video too short: %s (%d frames)" % (path, total))
    n = min(nframes, total)
    n -= n % 2                          # temporal_patch_size = 2
    want = [int(i) for i in np.arange(0, total, total / float(n))[:n]]

    frames, got, idx, wi = [], [], 0, 0
    while wi < len(want):
        ok = cap.grab()
        if not ok:
            break
        if idx == want[wi]:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                got.append(idx)
            wi += 1
        idx += 1
    cap.release()
    # CAP_PROP_FRAME_COUNT can overstate the real length; pad with the last
    # decoded frame so the frame count stays even. 帧数虚报时用末帧补齐。
    while len(frames) < n and frames:
        frames.append(frames[-1])
        got.append(got[-1])
    if len(frames) < 2:
        raise IOError("decoded too few frames: %s" % path)
    frames = augment_frames(frames, augment)

    arr = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    h, w = frames[0].shape[:2]
    meta = VideoMetadata(total_num_frames=total, fps=float(fps), width=w,
                         height=h, duration=total / float(fps),
                         video_backend="opencv", frames_indices=got)
    return arr, meta


def _seek(cap, first):
    """Position `cap` on frame `first`; returns False if the video ends first.

    Seek, then confirm where it landed; keyframe-only seeking can miss, so fall
    back to a sequential grab from frame 0 (same guard as build_train_clips.py).
    先跳转再核对落点，跳不准就退回顺序解码。
    """
    if first <= 0:
        return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == first:
        return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(first):
        if not cap.grab():
            return False
    return True


def sample_video_steps(path, n_steps, start_sec=0.0, end_sec=None, augment=None):
    """STEP_FPS frames per one-second step of [start_sec, end_sec) -> (T,C,H,W, metadata).

    Frames of step k sit at start_sec + k + i / STEP_FPS. Metadata indices are
    relative to the window start, so the `<t seconds>` stamps count from the
    window start exactly as they do on the training clips.
    帧索引相对窗口起点，长视频分窗时时间戳与训练片段一致。
    """
    from transformers.video_utils import VideoMetadata

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first = max(0, min(total - 1, int(round(start_sec * fps))))
    last = total - 1
    if end_sec is not None:
        last = max(first, min(last, int(round(end_sec * fps)) - 1))
    want = [min(last, first + int(round((k + i / float(STEP_FPS)) * fps)))
            for k in range(n_steps) for i in range(STEP_FPS)]

    need = sorted(set(want))
    if not _seek(cap, first):
        cap.release()
        raise IOError("video ends before %.2f s: %s" % (start_sec, path))
    got, idx, ni = {}, first, 0
    while ni < len(need):
        if not cap.grab():
            break
        if idx == need[ni]:
            ok, fr = cap.retrieve()
            if ok:
                got[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            ni += 1
        idx += 1
    cap.release()
    if not got:
        raise IOError("decoded no frames: %s" % path)
    # CAP_PROP_FRAME_COUNT can overstate the real length; a frame that could
    # not be decoded is replaced by the nearest earlier one. 帧数虚报时用前一帧。
    have = sorted(got)
    frames, rel = [], []
    for wi in want:
        j = wi if wi in got else max([h for h in have if h <= wi] or have[:1])
        frames.append(got[j])
        rel.append(j - first)
    frames = augment_frames(frames, augment)

    arr = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    h, w = frames[0].shape[:2]
    n = last - first + 1
    meta = VideoMetadata(total_num_frames=n, fps=float(fps), width=w, height=h,
                         duration=n / float(fps), video_backend="opencv",
                         frames_indices=rel)
    return arr, meta


def load_processor(model_path, max_pixels=DEFAULT_MAX_PIXELS,
                   min_pixels=DEFAULT_MIN_PIXELS):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path)
    vp = proc.video_processor
    vp.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    # Qwen3-VL / Qwen3.5 read `size` as a budget for the WHOLE clip (T*H*W),
    # Qwen2.5-VL as a per-frame one; build_inputs rescales it per clip.
    # Qwen3 系按整段视频总像素计，build_inputs 按帧数换算。
    proc._frame_pixels = (min_pixels, max_pixels)
    proc._total_budget = type(vp).__name__.startswith("Qwen3")
    vp.do_sample_frames = False
    if hasattr(vp, "cap_pixels_per_frame"):
        vp.cap_pixels_per_frame = False
    return proc


def load_model(model_path, adapter=None, quant_vision=False, for_training=False,
               quant=True):
    """Qwen2.5-VL / Qwen3.5 (+ optional LoRA adapter).

    quant=True : 4-bit NF4 LLM (the 8 GB recipe).
    quant=False: plain bf16 (24 GB+ cards, e.g. a 5090); an adapter trained
                 this way should also be served this way. bf16 全精度加载。
    """
    from transformers import (AutoConfig, AutoModelForImageTextToText,
                              BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration)

    kw = {}
    if quant:
        skip = ["lm_head"] + ([] if quant_vision else ["visual"])
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip)
    model_type = AutoConfig.from_pretrained(model_path).model_type
    cls = (Qwen2_5_VLForConditionalGeneration if model_type == "qwen2_5_vl"
           else AutoModelForImageTextToText)
    model = cls.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", **kw)
    model.config.use_cache = not for_training
    for p in model.parameters():
        p.requires_grad_(False)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=for_training)
    return model


def _apply_template(processor, messages, add_generation_prompt):
    # Qwen3.5 thinks by default; the task wants the bare JSON answer.
    # Templates without the flag ignore it. Qwen3.5 默认思考，这里关掉。
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False)


def build_inputs(processor, video_path, duration, target=None,
                 nframes=DEFAULT_NFRAMES, task="segments", start_sec=0.0, augment=None):
    """Tokenized model inputs for one clip. 构造单条样本的模型输入。

    With `target` (the assistant string) the returned dict also carries
    `n_target`: the number of trailing tokens that are supervised (answer +
    <|im_end|>). Without it, the text ends at the generation prompt.

    task="segments": `nframes` uniform frames, interval JSON prompt.
    task="scores"  : two frames per second of [start_sec, start_sec + duration),
                     one-digit-per-second prompt (`nframes` unused).
    augment        : training only, see augment_frames().
    """
    if task == "scores":
        frames, meta = sample_video_steps(video_path, score_steps(duration),
                                          start_sec, start_sec + duration, augment)
        prompt = render_score_prompt(duration)
    else:
        frames, meta = sample_video(video_path, nframes, augment)
        prompt = render_prompt(duration)
    if getattr(processor, "_total_budget", False):
        lo, hi = processor._frame_pixels
        n = frames.shape[0]
        processor.video_processor.size = {"shortest_edge": lo * n, "longest_edge": hi * n}
    user = {"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": prompt},
    ]}
    messages = [user]
    prompt_text = _apply_template(processor, messages, True)
    text = prompt_text
    if target is not None:
        # prompt + answer + <|im_end|>\n, exactly what the chat template would
        # render for the assistant turn. 与模板渲染的助手轮一致。
        text = prompt_text + target + "<|im_end|>\n"

    inputs = processor(text=[text], videos=[frames], video_metadata=[meta],
                       return_tensors="pt")
    if target is not None:
        tok = processor.tokenizer
        # Supervise answer + <|im_end|>; the trailing "\n" is not a target.
        # 监督答案与 <|im_end|>，末尾换行不监督。
        tgt_ids = tok(target + "<|im_end|>", add_special_tokens=False)["input_ids"]
        n_target = len(tgt_ids)
        n_tail = len(tok("\n", add_special_tokens=False)["input_ids"])
        # The answer must tokenize the same alone as in context, or the loss
        # would supervise the wrong tokens. 单独分词须与上下文中一致。
        ids = inputs["input_ids"][0].tolist()
        if ids[len(ids) - n_tail - n_target:len(ids) - n_tail] != tgt_ids:
            raise ValueError("target tokenization differs in context: %r" % target)
        inputs["n_target"] = n_target
        inputs["n_tail"] = n_tail
    return inputs
