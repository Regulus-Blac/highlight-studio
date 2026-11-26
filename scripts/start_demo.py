"""Start the saved GPU connection using OpenSSH; no model downloads or passwords in files."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import threading
import webbrowser

try:
    from .ssh_http_relay import drain_response
except ImportError:
    from ssh_http_relay import drain_response

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config' / 'demo.json'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/assets/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/assets/styles.css': ('styles.css', 'text/css; charset=utf-8')}


def load_profile(path=DEFAULT_CONFIG):
    profile = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(profile, dict):
        raise ValueError('Connection configuration must be an object')
    for field in ('host', 'user'):
        if not isinstance(profile.get(field), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', profile[field]):
            raise ValueError('Invalid connection field: ' + field)
    for field in ('ssh_port', 'remote_port', 'local_port'):
        if type(profile.get(field)) is not int or not 1 <= profile[field] <= 65535:
            raise ValueError('Invalid connection port: ' + field)
    for field in ('remote_root', 'remote_python'):
        value = profile.get(field)
        if not isinstance(value, str) or not value.startswith('/') or any(ord(char) < 32 for char in value):
            raise ValueError('Invalid remote path: ' + field)
    alias = profile.get('ssh_alias')
    if alias is not None and (not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', alias)):
        raise ValueError('Invalid SSH alias')
    return profile


def find_ssh():
    if os.name == 'nt':
        for executable in ('C:/Program Files/OpenSSH/ssh.exe', 'C:/Windows/System32/OpenSSH/ssh.exe'):
            if Path(executable).is_file():
                return executable
    executable = shutil.which('ssh')
    if not executable:
        raise RuntimeError('Please install OpenSSH before starting the demo')
    return executable


def ssh_target(profile, executable):
    alias = profile.get('ssh_alias')
    if alias:
        result = subprocess.run([executable, '-G', alias], capture_output=True, text=True, timeout=10, check=True)
        settings = dict(line.split(None, 1) for line in result.stdout.splitlines() if ' ' in line)
        if settings.get('hostname') != alias:
            actual = (settings.get('hostname'), settings.get('user'), settings.get('port'))
            expected = (profile['host'], profile['user'], str(profile['ssh_port']))
            if actual != expected:
                raise ValueError('The saved SSH alias differs from config/demo.json; update the saved configuration')
            return [alias]
    return ['-p', str(profile['ssh_port']), '-l', profile['user'], profile['host']]


def remote_command(profile):
    return (f"cd {shlex.quote(profile['remote_root'])} && bash app/scripts/demo_service.sh real 1>&2 && "
            f"exec {shlex.quote(profile['remote_python'])} -u app/scripts/ssh_http_relay.py --port {profile['remote_port']}")


def make_server(profile, process):
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != origin.removeprefix('http://') or self.headers.get('Origin', origin) != origin:
                self.send_error(403)
                return
            if self.command == 'GET' and self.path in ASSETS:
                filename, content_type = ASSETS[self.path]
                body = (ROOT / 'apps' / 'frontend' / filename).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.command == 'GET' and self.path == '/api/v1/deployment':
                body = json.dumps({'location': 'remote', 'mode': 'real', 'transport': 'ssh'}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= 512 * 1024 * 1024:
                self.send_error(413)
                return
            with lock:
                try:
                    if process.poll() is not None:
                        self.send_error(502, 'The connection ended; restart the demo')
                        return
                    head = {'method': self.command, 'path': self.path, 'headers': dict(self.headers), 'length': length}
                    process.stdin.write((json.dumps(head) + '\n').encode())
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 49152))
                        if not chunk:
                            raise EOFError('Upload interrupted')
                        process.stdin.write(base64.b64encode(chunk) + b'\n')
                        remaining -= len(chunk)
                    process.stdin.flush()
                    response = json.loads(process.stdout.readline(65537))
                    disconnected = False
                    try:
                        self.send_response(response['status'])
                        for key, value in response['headers'].items():
                            self.send_header(key, value)
                        self.end_headers()
                    except OSError:
                        disconnected = True
                    drain_response(process.stdout, None if disconnected else self.wfile, response['length'])
                except Exception as error:
                    if process.poll() is None:
                        process.terminate()
                    self.close_connection = True
                    print(f'Connection interrupted ({type(error).__name__}); restart the demo.', flush=True)

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = handle_request

    return ThreadingHTTPServer(('127.0.0.1', profile['local_port']), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    profile = load_profile(args.config)
    ssh = find_ssh()
    command = [ssh, '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
               *ssh_target(profile, ssh), remote_command(profile)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    server = None
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(process.stdout.readline(65537)), daemon=True).start()
        try:
            marker = ready.get(timeout=240).strip()
        except queue.Empty as error:
            raise RuntimeError('The remote service did not start within four minutes') from error
        if marker != b'HIGHLIGHT_RELAY_READY':
            raise RuntimeError('Cannot connect. Check the saved SSH login and remote service.')
        server = make_server(profile, process)
        address = f"http://127.0.0.1:{profile['local_port']}"
        print('Ready: ' + address, flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


if __name__ == '__main__':
    main()
