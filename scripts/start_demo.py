"""Start the local test page immediately; connect the saved remote in the background."""
import argparse
import base64
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import webbrowser

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.backend.server import make_server as make_local_server
from services.model_adapter.adapter import MockAdapter
from services.media.probe import MediaError, binary
from services.task_service.store import TaskError

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


def stop_process(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for stream in (process.stdin, process.stdout):
        if stream:
            stream.close()


class RemoteConnection:
    def __init__(self, profile, *, config_path=None, process=None, startup_timeout=240, local_demo=True):
        self.profile, self.config_path = profile, config_path
        self.process, self.startup_timeout = process, startup_timeout
        self.local_demo = local_demo
        self.lock, self.transfer_lock = threading.Lock(), threading.Lock()
        self.closed, self.thread = threading.Event(), None
        self._status = 'connected' if process else 'unavailable'

    def status(self):
        with self.lock:
            if self._status == 'connected' and (not self.process or self.process.poll() is not None):
                self._status = 'unavailable'
            return self._status

    def descriptor(self):
        result = {'location': 'remote', 'mode': 'real', 'transport': 'ssh'}
        if self.local_demo:
            result.update(connection=self.status(), local_demo=True)
        return result

    def connect(self):
        if self.status() == 'connected':
            return
        with self.lock:
            if self.closed.is_set() or self.thread and self.thread.is_alive():
                return
            previous, self.process = self.process, None
            stop_process(previous)
            self._status = 'connecting'
            self.thread = threading.Thread(target=self._connect, name='demo-remote-connect', daemon=True)
            self.thread.start()

    def _start_process(self):
        profile = load_profile(self.config_path) if self.config_path else self.profile
        ssh = find_ssh()
        command = [ssh, '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                   '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
                   *ssh_target(profile, ssh), remote_command(profile)]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _connect(self):
        process, connected = None, False
        try:
            process = self._start_process()
            with self.lock:
                if self.closed.is_set():
                    return
                self.process = process
            ready = queue.Queue()
            reader = threading.Thread(target=lambda: ready.put(process.stdout.readline(65537)), daemon=True)
            reader.start()
            marker = ready.get(timeout=self.startup_timeout).strip()
            if marker != b'HIGHLIGHT_RELAY_READY' or process.poll() is not None:
                raise RuntimeError('SSH relay did not become ready')
            with self.lock:
                if not self.closed.is_set():
                    self._status, connected = 'connected', True
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, queue.Empty) as error:
            detail = 'Timed out waiting for remote service' if isinstance(error, queue.Empty) else str(error)
            print(f'Remote connection unavailable ({type(error).__name__}): {detail}. Local test page remains available.', file=sys.stderr, flush=True)
        finally:
            if not connected:
                with self.lock:
                    if self.process is process:
                        self.process = None
                    self._status = 'unavailable'
                stop_process(process)

    def invalidate(self, process):
        with self.lock:
            if self.process is process:
                self.process = None
                self._status = 'unavailable'
        stop_process(process)

    def close(self):
        self.closed.set()
        with self.lock:
            process, self.process = self.process, None
            self._status = 'unavailable'
        stop_process(process)
        if self.thread:
            self.thread.join(timeout=5)


def demo_adapter():
    # Media simulation is optional; missing CPU tools must not block the page.
    try:
        from PIL import Image  # noqa: F401
        from services.model_adapter.development import SyntheticMediaAdapter
        binary('ffmpeg'); binary('ffprobe')
        return SyntheticMediaAdapter()
    except (ImportError, OSError, MediaError):
        return MockAdapter()


def make_server(profile, process, *, runtime=None, adapter=None):
    connection = process if isinstance(process, RemoteConnection) else RemoteConnection(profile, process=process, local_demo=False)
    server = make_local_server(port=profile['local_port'], runtime=runtime or ROOT/'runtime/gpu-demo/local',
                               adapter=adapter or MockAdapter(), allow_mock_media=True)
    base = server.RequestHandlerClass

    class Handler(base):
        def handle_request(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != origin.removeprefix('http://') or self.headers.get('Origin', origin) != origin:
                self.error(403, 'origin_not_allowed', 'Only same-origin requests are allowed')
                return
            if self.path.startswith('/demo/') and connection.local_demo:
                self.path = self.path[len('/demo'):]
                method = getattr(super(), 'do_'+self.command, None)
                if method:
                    method()
                else:
                    self.error(405, 'method_not_allowed', 'Method is unavailable')
                return
            if self.command == 'GET' and (self.path in ASSETS or self.path == '/health'):
                super().do_GET()
                return
            if self.command == 'GET' and self.path == '/api/v1/deployment':
                self.respond(200, connection.descriptor())
                return
            if self.command == 'POST' and self.path == '/api/v1/deployment' and connection.local_demo:
                try:
                    if self.read_json() != {}:
                        raise ValueError('Reconnect accepts an empty object only')
                except (ValueError, TaskError):
                    self.error(400, 'invalid_request', 'Reconnect accepts an empty JSON object')
                    return
                connection.connect()
                self.respond(200, connection.descriptor())
                return
            if connection.status() != 'connected':
                if self.command == 'GET' and self.path == '/api/v1/model':
                    self.respond(200, {'api_version': '1', 'engine': 'remote-unavailable', 'mock': False,
                                       'available': False, 'prediction_api_version': '1'})
                elif self.command == 'GET' and self.path == '/ready':
                    self.respond(503, {'ready': False, 'engine': 'remote-unavailable', 'mock': False})
                else:
                    self.error(503, 'remote_unavailable', 'Remote server is not connected; local demo remains available')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= 512 * 1024 * 1024:
                self.send_error(413)
                return
            with connection.transfer_lock:
                process = connection.process
                responded = False
                try:
                    if process is None or process.poll() is not None:
                        self.error(503, 'remote_unavailable', 'Remote server is not connected')
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
                        responded = True
                    except OSError:
                        disconnected = True
                    drain_response(process.stdout, None if disconnected else self.wfile, response['length'])
                except Exception as error:
                    connection.invalidate(process)
                    if not responded:
                        self.error(503, 'remote_unavailable', 'Remote connection ended; local demo remains available')
                    self.close_connection = True
                    print(f'Remote connection interrupted ({type(error).__name__}). Local page remains available.', flush=True)

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = handle_request

    server.RequestHandlerClass = Handler
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--runtime', type=Path, default=ROOT/'runtime/gpu-demo/local', help='Separate local demonstration data')
    args = parser.parse_args()
    profile = load_profile(args.config)
    connection = RemoteConnection(profile, config_path=args.config)
    server = None
    try:
        server = make_server(profile, connection, runtime=args.runtime, adapter=demo_adapter())
        connection.connect()
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
        connection.close()


if __name__ == '__main__':
    main()
