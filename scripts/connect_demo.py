"""Open the remote demo locally through SSH, including Suanjia's instance menu.

Credentials are requested interactively and never written to disk or sent to a browser.
Install requirements-demo.txt; see docs/demo-runbook.md.
"""
import argparse
import base64
import getpass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import threading
import time
import webbrowser

import paramiko
from ssh_http_relay import drain_response

class ConfirmHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        import base64, hashlib
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        print(f'New SSH host {hostname}: SHA256:{fingerprint}')
        if input('Trust this host? Type yes: ').strip() != 'yes':
            raise RuntimeError('Host was not trusted')
        client.get_host_keys().add(hostname, key.get_name(), key)

def wait_for(channel, marker, timeout=120):
    data = b''; deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if channel.recv_ready():
            chunk = channel.recv(65536)
            if not chunk: raise EOFError('SSH disconnected')
            data += chunk
            if b'HIGHLIGHT_START_FAILED' in data and b'\nHIGHLIGHT_START_FAILED' in data:
                raise RuntimeError('Remote startup failed: '+data.decode(errors='replace')[-3000:])
            if marker in data: return data
            if len(data) > 200000: data = data[-100000:]
        elif channel.closed:
            raise RuntimeError('Remote startup failed: '+data.decode(errors='replace')[-4000:])
        else: time.sleep(.1)
    raise TimeoutError('Remote startup timed out: '+data.decode(errors='replace')[-4000:])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True); parser.add_argument('--user', required=True)
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--instance', type=int, help='Instance index for the gateway menu')
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--remote-python', default='/usr/bin/python3', help='Remote Python interpreter for the HTTP relay')
    parser.add_argument('--mode', choices=['mock','real'], default='real')
    parser.add_argument('--port', type=int, default=18000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    ssh = paramiko.SSHClient(); ssh.load_system_host_keys()
    known = Path.home()/'.ssh/known_hosts'
    if known.exists(): ssh.load_host_keys(str(known))
    ssh.set_missing_host_key_policy(ConfirmHost())
    ssh.connect(args.host, port=args.ssh_port, username=args.user, password=getpass.getpass('SSH password: '), look_for_keys=False, allow_agent=False, timeout=20)
    ssh.get_transport().set_keepalive(20)
    channel = ssh.invoke_shell(width=200, height=40)
    if args.instance is not None:
        wait_for(channel, '输入服务器序号>'.encode()); channel.sendall(f'{args.instance}\r'); time.sleep(1); channel.sendall('\r')
        wait_for(channel, b'# ')
    # A marker command executes only after the gateway has entered the instance shell.
    channel.sendall("printf '\\nHIGHLIGHT_SHELL_%s\\n' READY\n")
    wait_for(channel, b'HIGHLIGHT_SHELL_READY\r\n')
    root = shlex.quote(args.remote_root)
    remote_port = 8001 if args.mode == 'mock' else 8002
    command = (f'cd {root} && bash app/scripts/demo_service.sh {args.mode} && '
               f'stty raw -echo && exec {shlex.quote(args.remote_python)} -u app/scripts/ssh_http_relay.py --port {remote_port}')
    channel.sendall(command+" || printf '\\nHIGHLIGHT_START_%s\\n' FAILED\n")
    try: wait_for(channel, b'HIGHLIGHT_RELAY_READY\n', timeout=240)
    except Exception:
        channel.close(); ssh.close(); raise
    channel.settimeout(180)
    source = channel.makefile('rb'); sink = channel.makefile('wb'); lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            # Bind only loopback and reject cross-origin browser writes / DNS rebinding.
            origin = f'http://127.0.0.1:{args.port}'
            if self.headers.get('Host') != f'127.0.0.1:{args.port}' or self.headers.get('Origin', origin) != origin:
                self.send_error(403); return
            if self.path == '/api/v1/deployment':
                body = json.dumps({'location':'remote','mode':args.mode,'transport':'ssh'}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            length = int(self.headers.get('Content-Length','0'))
            if not 0 <= length <= 512*1024*1024:
                self.send_error(413);return
            with lock:
                try:
                    head = {'method': self.command, 'path': self.path, 'headers': dict(self.headers), 'length': length}
                    sink.write((json.dumps(head)+'\n').encode())
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining,49152))
                        if not chunk: raise EOFError('Browser upload interrupted')
                        sink.write(base64.b64encode(chunk)+b'\n'); remaining -= len(chunk)
                    sink.flush()
                    response = json.loads(source.readline(65537))
                    disconnected = False
                    try:
                        self.send_response(response['status'])
                        for key,value in response['headers'].items(): self.send_header(key,value)
                        self.end_headers()
                    except OSError:
                        disconnected = True
                    drain_response(source, None if disconnected else self.wfile, response['length'])
                except Exception as error:
                    # Framing is no longer safe after an interrupted exchange. Reconnect explicitly.
                    channel.close()
                    self.close_connection = True
                    print(f'Demo connection interrupted ({type(error).__name__}). Restart this connector to reconnect.', flush=True)
        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = handle_request
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Demo ready: http://127.0.0.1:{args.port} ({args.mode}, SSH encrypted). Keep this window open.',flush=True)
    if not args.no_browser: webbrowser.open(f'http://127.0.0.1:{args.port}')
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); channel.close(); ssh.close()

if __name__ == '__main__': main()
