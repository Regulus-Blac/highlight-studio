"""CI-only private submodule checkout using a dedicated read-only deploy key."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import urllib.request

from model_ref import MODULE, ROOT, git


key_value = os.environ.get("COMPETITION_READ_KEY")
if not key_value:
    raise SystemExit("Missing Actions secret COMPETITION_READ_KEY (read-only upstream deploy key)")
api_token = os.environ.get("GITHUB_TOKEN", "").strip()
if not api_token:
    raise SystemExit("Missing Actions token GITHUB_TOKEN (required for authenticated GitHub metadata)")
directory = Path(os.environ["RUNNER_TEMP"]) / "competition-ssh"
directory.mkdir(mode=0o700, exist_ok=True)
key = directory / "id_ed25519"
key.write_text(key_value.strip() + "\n", encoding="utf-8")
key.chmod(0o600)
# Fetch published host keys over verified HTTPS, then require an exact SSH match.
request = urllib.request.Request("https://api.github.com/meta", headers={
    "User-Agent": "highlight-studio-ci",
    "Accept": "application/vnd.github+json",
    "Authorization": "Bearer " + api_token,
})
with urllib.request.urlopen(request, timeout=30) as response:
    host_keys = json.load(response)["ssh_keys"]
known_hosts = directory / "known_hosts"
known_hosts.write_text("".join("github.com " + item + "\n" for item in host_keys), encoding="utf-8")
ssh_command = "ssh -i " + shlex.quote(str(key)) + " -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=" + shlex.quote(str(known_hosts))
rewrite = "url.ssh://git@github.com/.insteadOf=https://github.com/"
subprocess.run(
    ["git", "-c", rewrite, "submodule", "update", "--init", "--recursive"],
    cwd=ROOT, env={**os.environ, "GIT_SSH_COMMAND": ssh_command}, check=True,
)
module = ROOT / MODULE
git("config", "core.sshCommand", ssh_command, cwd=module)
git("config", "url.ssh://git@github.com/.insteadOf", "https://github.com/", cwd=module)
print("Private competition submodule initialized with dedicated read-only deploy key")
