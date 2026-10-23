"""CPU-only integrity checks; does not import the research environment."""

import json
import subprocess

from model_ref import MODULE, ROOT, config, git


def main():
    lock = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))
    module = ROOT / MODULE
    # rev-parse alone can accidentally resolve to the parent when a submodule is missing.
    if not (module / ".git").is_file():
        raise SystemExit("Submodule is missing. Run python scripts/bootstrap.py")
    checks = {
        "lock path": lock["path"] == MODULE,
        "lock branch": lock["branch"] == config("branch"),
        "lock repository": lock["repository"] == config("url"),
        "remote URL": git("config", "--get", "remote.origin.url", cwd=module) == lock["repository"],
        "lock commit": git("rev-parse", "HEAD", cwd=module) == lock["commit"],
        "upstream checkout clean": not git("status", "--porcelain", "--untracked-files=all", cwd=module),
        "upstream license": (module / "LICENSE").is_file(),
        "baseline available": (module / "baseline" / "README.md").is_file(),
    }
    # A gitlink must be an actual submodule; staged upgrades must match the lock.
    entry = git("ls-files", "--stage", "--", MODULE).split()
    checks["gitlink exists"] = len(entry) >= 3 and entry[0] == "160000"
    if len(entry) >= 3:
        old_lock = subprocess.run(["git", "show", ":model.lock.json"], cwd=ROOT, capture_output=True, text=True)
        if old_lock.returncode == 0:
            staged_lock = json.loads(old_lock.stdout)
            checks["staged reference consistent"] = entry[1] == staged_lock["commit"]
    # No network is required here; fresh recursive clones prove remote availability.
    refs = git("for-each-ref", "--format=%(refname)", "--contains", lock["commit"], "refs/remotes/", cwd=module)
    checks["commit belongs to fetched upstream history"] = bool(refs)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
