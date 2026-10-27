"""Controlled media streaming with single byte Range support for browser seeking."""

import re

from services.task_service.store import TaskError


def send_file(handler, path, content_type, digest):
    if not path.is_file():
        raise TaskError(404, "artifact_not_found", "Stored media file is unavailable")
    total = path.stat().st_size
    start, end, status = 0, total - 1, 200
    value = handler.headers.get("Range")
    if value:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
        if not match or not any(match.groups()):
            start = total
        elif not match[1]:
            start = max(0, total - int(match[2])) if int(match[2]) else total
        else:
            start = int(match[1])
            end = min(total - 1, int(match[2])) if match[2] else total - 1
        if start >= total or start > end:
            handler.send_response(416)
            handler.cors()
            handler.send_header("Content-Range", f"bytes */{total}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        status = 206
    handler.send_response(status)
    handler.cors()
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("ETag", '"' + digest + '"')
    handler.send_header("Cache-Control", "private, no-cache")
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{total}")
    handler.send_header("Content-Length", str(end - start + 1))
    handler.end_headers()
    with path.open("rb") as source:
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)
