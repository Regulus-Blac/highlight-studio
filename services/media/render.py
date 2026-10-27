"""Bounded-memory CPU reference renderer. No model imports and no shell commands."""

import json
from pathlib import Path
import subprocess
import threading
import time

from services.media.probe import MediaError, binary, probe


def render_jsonl(result, plan, output):
    rendered = {**result, "predictions": [{"frame": f["frame"], "bboxes": f["crop"][:3]} for f in plan["frame_crops"]]}
    Path(output).write_text(json.dumps(rendered, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def render_mp4(source, plan, output, *, timeout=900, cancel_event=None):
    try:
        from PIL import Image
    except ImportError as error:
        raise MediaError("media_tools_unavailable", "Install requirements-media.txt for the CPU renderer") from error
    if not plan["frame_crops"]:
        raise MediaError("no_enabled_segments", "An MP4 requires at least one enabled source frame")
    ffmpeg = binary("ffmpeg")
    meta, size = plan["metadata"], plan["output"]
    fps = f"{meta['fps_num']}/{meta['fps_den']}"
    segments = [s for s in plan["segments"] if s["enabled"]]
    selection = "+".join(f"between(n,{s['start_frame']},{s['end_frame_exclusive']-1})" for s in segments)
    # Filter expressions go into a generated file, never into a shell or a URL.
    work = Path(output).parent
    (work / "select.ffscript").write_text(f"select='{selection}'", encoding="utf-8")
    common = [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist", "file,pipe"]
    decode = [*common, "-noautorotate", "-i", str(source), "-map", "0:v:0", "-an", "-filter_script:v", str(work / "select.ffscript"), "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    encode = [*common, "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{size['width']}x{size['height']}", "-framerate", fps, "-i", "pipe:0"]
    if meta["has_audio"]:
        encode += ["-i", str(source)]
        filters = []
        for i, segment in enumerate(segments):
            start = segment["start_frame"] * meta["fps_den"] / meta["fps_num"]
            end = segment["end_frame_exclusive"] * meta["fps_den"] / meta["fps_num"]
            filters.append(f"[1:a:0]atrim=start={start:.12f}:end={end:.12f},asetpts=PTS-STARTPTS,apad,atrim=duration={end-start:.12f}[a{i}]")
        filters.append("".join(f"[a{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=0:a=1[a]")
        (work / "audio.ffscript").write_text(";\n".join(filters), encoding="utf-8")
        encode += ["-filter_complex_threads", "1", "-filter_complex_script", str(work / "audio.ffscript"), "-map", "0:v:0", "-map", "[a]", "-c:a", "aac"]
    else:
        encode += ["-map", "0:v:0", "-an"]
    encode += ["-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-t", str(plan["duration_seconds"]), "-movflags", "+faststart", "-f", "mp4", "-y", str(output)]
    processes = []
    expired = threading.Event()
    interrupted = threading.Event()
    done = threading.Event()

    def terminate():
        for process in processes:
            if process.poll() is None:
                process.kill()

    def watch():
        deadline = time.monotonic() + timeout
        while not done.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                interrupted.set()
                terminate()
                return
            if time.monotonic() >= deadline:
                expired.set()
                terminate()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    with (work / "ffmpeg.log").open("wb") as log:
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=log)
            processes.append(decoder)
            encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            processes.append(encoder)
            watcher.start()
            length = meta["width"] * meta["height"] * 3
            for item in plan["frame_crops"]:
                data = decoder.stdout.read(length)
                if len(data) != length:
                    raise MediaError("render_failed", "Decoder returned fewer frames than the frozen render plan")
                image = Image.frombytes("RGB", (meta["width"], meta["height"]), data)
                x, y, width, height = item["crop"]
                image = image.crop((x, y, x + width, y + height))
                if image.size != (size["width"], size["height"]):
                    image = image.resize((size["width"], size["height"]), Image.Resampling.LANCZOS)
                encoder.stdin.write(image.tobytes())
            if decoder.stdout.read(1):
                raise MediaError("render_failed", "Decoder returned extra frames")
            encoder.stdin.close()
            if decoder.wait() or encoder.wait():
                raise MediaError("render_failed", "FFmpeg failed; inspect the private worker log")
        except (OSError, subprocess.SubprocessError, MediaError) as error:
            if interrupted.is_set():
                raise MediaError("render_interrupted", "Export worker is stopping; retry with a new key") from error
            if expired.is_set():
                raise MediaError("render_timeout", "CPU rendering exceeded the configured timeout") from error
            if isinstance(error, MediaError):
                raise
            raise MediaError("render_failed", "FFmpeg could not complete the render") from error
        finally:
            done.set()
            if watcher.ident:
                watcher.join(timeout=1)
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream and not stream.closed:
                        try:
                            stream.close()
                        except OSError:
                            pass
    actual = probe(output)
    if any(actual[k] != value for k, value in {"width": size["width"], "height": size["height"], "frame_count": len(plan["frame_crops"]), "has_audio": meta["has_audio"]}.items()):
        raise MediaError("render_failed", "Rendered file failed dimensions, frame count or audio verification")
    return actual
