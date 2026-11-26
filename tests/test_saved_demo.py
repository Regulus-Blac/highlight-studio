"""Saved connections and the real HTTP relay boundary, without GPU or SSH login."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.start_demo import DEFAULT_CONFIG, load_profile, make_server, remote_command, ssh_target

ROOT = Path(__file__).resolve().parents[1]


class SavedConnectionTests(unittest.TestCase):
    def test_default_connection_needs_no_command_line_address(self):
        profile = load_profile()
        self.assertTrue(profile['host'])
        self.assertTrue(profile['remote_root'].startswith('/'))
        result = subprocess.run([sys.executable, str(ROOT/'scripts/start_demo.py'), '--help'], capture_output=True, check=True, text=True)
        self.assertIn('--config', result.stdout)
        self.assertNotIn('--host', result.stdout)
        self.assertNotIn('password', profile)

    def test_invalid_ports_and_remote_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'connection.json'
            for change in ({'local_port': True}, {'ssh_port': 0}, {'host': '-oProxyCommand=bad'}, {'remote_root': '/tmp\nfalse'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**load_profile(DEFAULT_CONFIG), **change}), encoding='utf-8')
                    load_profile(path)

    def test_alias_cannot_silently_select_a_different_server(self):
        profile = load_profile()
        result = subprocess.CompletedProcess([], 0, 'hostname other.example\nuser root\nport 22\n', '')
        with patch('scripts.start_demo.subprocess.run', return_value=result), self.assertRaises(ValueError):
            ssh_target(profile, 'ssh')
        profile['remote_root'] = '/data/demo with spaces'
        self.assertIn("cd '/data/demo with spaces'", remote_command(profile))


@contextmanager
def relay_service():
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            self.send_response(201)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = bytes(range(256)) * 400
            partial = bool(self.headers.get('Range'))
            if partial:
                body = body[-16:]
            self.send_response(206 if partial else 200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    process = subprocess.Popen([sys.executable, str(ROOT/'scripts/ssh_http_relay.py'), '--port', str(upstream.server_port)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout.readline() == b'HIGHLIGHT_RELAY_READY\n'
    proxy = make_server({**load_profile(), 'local_port': 0}, process)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{proxy.server_port}'
    finally:
        proxy.shutdown();proxy.server_close();thread.join(5)
        process.stdin.close();process.wait(timeout=5);process.stdout.close();process.stderr.close()
        upstream.shutdown();upstream.server_close();upstream_thread.join(5)


class SavedDemoHTTPTests(unittest.TestCase):
    def test_local_page_and_remote_binary_transfer_share_one_origin(self):
        with relay_service() as base:
            with urlopen(base) as response:
                self.assertIn(b'Highlight Studio', response.read())
            with urlopen(base+'/api/v1/deployment') as response:
                self.assertEqual(json.load(response), {'location':'remote','mode':'real','transport':'ssh'})
            payload = bytes(range(256)) * 1000
            with urlopen(Request(base+'/api/v1/test-upload', data=payload)) as response:
                self.assertEqual(response.status, 201)
                self.assertEqual(hashlib.sha256(response.read()).digest(), hashlib.sha256(payload).digest())
            with urlopen(Request(base+'/api/v1/test-video', headers={'Range':'bytes=-16'})) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), bytes(range(240,256)))
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/api/v1/test-upload', data=b'test', headers={'Origin':'https://other.example'}))
            self.assertEqual(error.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
