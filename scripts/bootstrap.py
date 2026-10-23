"""Initialize this clone and install local safeguards; no GPU dependencies."""

import sys

from model_ref import MODULE, ROOT, git


if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required")
git("submodule", "sync", "--recursive")
git("submodule", "update", "--init", "--recursive")
git("config", "core.hooksPath", ".githooks")
git("config", "fetch.recurseSubmodules", "on-demand")
git("config", "push.recurseSubmodules", "no")
git("config", "diff.submodule", "log")
module = ROOT / MODULE
hooks = ROOT / ".githooks" / "competition"
git("config", "core.hooksPath", hooks.as_posix(), cwd=module)
git("config", "remote.origin.pushurl", "https://invalid.invalid/competition-push-disabled", cwd=module)
for path in (ROOT / ".githooks").rglob("*"):
    if path.is_file():
        path.chmod(path.stat().st_mode | 0o111)
print("Ready. Competition checkout has local commit/push safeguards; original repository is untouched.")
print("Run: python scripts/check_repository.py; python scripts/smoke.py; python -m apps.backend.server")
