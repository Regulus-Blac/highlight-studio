"""Upload an owned video and exercise the stable D/E task API."""

import argparse
import http.client
import json
from pathlib import Path
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ratio", default="9:16")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    ratio = [int(x) for x in args.ratio.split(":")]
    if len(ratio) != 2 or min(ratio) <= 0:
        parser.error("--ratio must be two positive integers such as 9:16")
    base = args.base_url.rstrip("/")
    url = urlsplit(base)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.netloc, timeout=60)
    with args.video.open("rb") as video:
        connection.request("POST", url.path + "/api/v1/uploads?filename=" + quote(args.video.name), body=video, headers={"Content-Type": "application/octet-stream", "Content-Length": str(args.video.stat().st_size)})
        response = connection.getresponse()
        upload = json.loads(response.read())
        if response.status != 201:
            raise SystemExit(json.dumps(upload, ensure_ascii=False))
    connection.close()

    def request(path, payload=None):
        req = Request(base + path, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Idempotency-Key": key})
        with urlopen(req, timeout=30) as response:
            return json.load(response)

    key = uuid4().hex
    task = request("/api/v1/tasks", {"api_version": "1", "upload_id": upload["upload_id"], "target_ratio": ratio})
    deadline = time.monotonic() + args.timeout
    previous = None
    while True:
        if task["status"] != previous:
            print(f"{task['task_id']} {task['status']} engine={task['engine']} mock={task['mock']}", flush=True)
            previous = task["status"]
        if task["status"] in {"succeeded", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("Polling timed out; task remains available through the query API")
        time.sleep(0.3)
        task = request("/api/v1/tasks/" + task["task_id"])
    print(json.dumps(task["result"] or task["error"], ensure_ascii=False, indent=2))
    if task["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
