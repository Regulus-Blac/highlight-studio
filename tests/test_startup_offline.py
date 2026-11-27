"""A missing remote must not prevent local HTTP startup or mix task stores."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.start_demo import RemoteConnection, load_profile, make_server
from services.model_adapter.adapter import MockAdapter


@contextmanager
def gateway(connection):
    with tempfile.TemporaryDirectory() as directory:
        server = make_server({**load_profile(), 'local_port': 0}, connection, runtime=Path(directory), adapter=MockAdapter())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}'
        finally:
            connection.close()
            server.shutdown(); server.server_close(); thread.join(5)


def request(base, path, body=None, headers=None):
    payload = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
    try:
        with urlopen(Request(base+path, data=payload, headers=headers or {}), timeout=3) as response:
            data = response.read()
            return response.status, json.loads(data) if 'application/json' in response.headers.get('Content-Type', '') else data
    except HTTPError as error:
        return error.code, json.load(error)


class OfflineStartupTests(unittest.TestCase):
    def test_page_is_available_while_connection_is_pending(self):
        entered, release = threading.Event(), threading.Event()
        connection = RemoteConnection(load_profile())
        def slow_connect():
            entered.set(); release.wait(5)
        with patch.object(connection, '_connect', side_effect=slow_connect), gateway(connection) as base:
            connection.connect(); self.assertTrue(entered.wait(1))
            self.assertEqual(request(base, '/')[0], 200)
            self.assertEqual(request(base, '/health')[1]['status'], 'ok')
            self.assertEqual(request(base, '/api/v1/deployment')[1]['connection'], 'connecting')
            self.assertEqual(request(base, '/ready')[0], 503)
            self.assertEqual(request(base, '/demo/ready')[0], 200)
            release.set()

    def test_failed_connection_keeps_local_tasks_and_blocks_remote_uploads(self):
        connection = RemoteConnection(load_profile())
        with patch.object(connection, '_start_process', side_effect=RuntimeError('SSH not installed')), gateway(connection) as base:
            connection.connect(); connection.thread.join(2)
            descriptor = request(base, '/api/v1/deployment')[1]
            self.assertEqual(descriptor['connection'], 'unavailable')
            self.assertTrue(descriptor['local_demo'])
            self.assertEqual(request(base, '/api/v1/model')[1]['available'], False)
            self.assertEqual(request(base, '/api/v1/uploads?filename=test.mp4', b'local bytes')[0], 503)
            status, upload = request(base, '/demo/api/v1/uploads?filename=test.mp4', b'local bytes')
            self.assertEqual(status, 201)
            status, task = request(base, '/demo/api/v1/tasks', {'api_version': '1', 'upload_id': upload['upload_id'], 'target_ratio': [16, 9]})
            self.assertEqual(status, 202)
            for _ in range(30):
                _, task = request(base, '/demo/api/v1/tasks/'+task['task_id'])
                if task['status'] == 'succeeded': break
                time.sleep(.05)
            self.assertEqual(task['status'], 'succeeded')
            self.assertTrue(task['mock'])
            self.assertEqual(request(base, '/demo/api/v1/tasks/'+task['task_id']+'/source'), (200, b'local bytes'))
            self.assertEqual(request(base, '/api/v1/tasks/'+task['task_id'])[0], 503)
            self.assertEqual(request(base, '/demo/api/v1/uploads?filename=test.mp4', b'bad', {'Origin': 'https://other.example'})[0], 403)

    def test_timeout_reaps_the_failed_connection_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        connection = RemoteConnection(load_profile(), startup_timeout=.05)
        with patch.object(connection, '_start_process', return_value=process), gateway(connection) as base:
            connection.connect(); connection.thread.join(3)
            self.assertFalse(connection.thread.is_alive())
            self.assertIsNotNone(process.poll())
            self.assertEqual(request(base, '/api/v1/deployment')[1]['connection'], 'unavailable')
            self.assertEqual(request(base, '/')[0], 200)

    def test_reconnect_and_disconnect_do_not_switch_local_api_namespace(self):
        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps({'remote': True}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(payload))); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(payload)
            def log_message(self, *args): pass
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True); thread.start()
        connection = RemoteConnection(load_profile())
        def relay():
            return subprocess.Popen([sys.executable, 'scripts/ssh_http_relay.py', '--port', str(upstream.server_port)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            with gateway(connection) as base:
                with patch.object(connection, '_start_process', side_effect=RuntimeError('Offline')):
                    connection.connect(); connection.thread.join(2)
                with patch.object(connection, '_start_process', side_effect=relay):
                    self.assertEqual(request(base, '/api/v1/deployment', {})[0], 200)
                    connection.thread.join(2)
                self.assertEqual(request(base, '/api/v1/deployment')[1]['connection'], 'connected')
                self.assertEqual(request(base, '/api/v1/model')[1], {'remote': True})
                self.assertTrue(request(base, '/demo/api/v1/model')[1]['mock'])
                connection.process.terminate(); connection.process.wait(3)
                self.assertEqual(request(base, '/api/v1/deployment')[1]['connection'], 'unavailable')
                self.assertEqual(request(base, '/')[0], 200)
                self.assertTrue(request(base, '/demo/ready')[1]['ready'])
        finally:
            upstream.shutdown(); upstream.server_close(); thread.join(5)


if __name__ == '__main__': unittest.main()
