"""Manage the upstream branch and reproducible gitlink without editing upstream."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODULE = "vendor/competition"


def git(*args, cwd=ROOT):
    # Git hooks export the parent's index/context, which must not leak into -C submodule calls.
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, env=env).strip()


def config(key):
    return git("config", "-f", ".gitmodules", "--get", f"submodule.{MODULE}.{key}")


def write_lock():
    lock = {
        "schema_version": 1,
        "repository": config("url"),
        "path": MODULE,
        "branch": config("branch"),
        "commit": git("rev-parse", "HEAD", cwd=ROOT / MODULE),
    }
    (ROOT / "model.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("update")
    switch = sub.add_parser("switch")
    switch.add_argument("branch")
    args = parser.parse_args()
    if args.command == "status":
        print((ROOT / "model.lock.json").read_text(encoding="utf-8"))
        print(git("submodule", "status", "--", MODULE))
        return
    module = ROOT / MODULE
    if not (module / ".git").is_file():
        git("submodule", "update", "--init", "--", MODULE)
    if git("status", "--porcelain", "--untracked-files=all", cwd=module):
        raise SystemExit("Competition checkout has changes. Preserve them in the original repository first.")
    branch = args.branch if args.command == "switch" else config("branch")
    git("check-ref-format", "refs/heads/" + branch)
    # Only update a remote-tracking ref; never change an ABC development branch.
    remote_ref = "refs/remotes/origin/" + branch
    git("fetch", "origin", "+refs/heads/" + branch + ":" + remote_ref, cwd=module)
    commit = git("rev-parse", remote_ref + "^{commit}", cwd=module)
    git("checkout", "--detach", commit, cwd=module)
    git("config", "-f", ".gitmodules", f"submodule.{MODULE}.branch", branch)
    write_lock()
    print("Review and commit .gitmodules, model.lock.json and vendor/competition in the product repository.")


if __name__ == "__main__":
    main()
