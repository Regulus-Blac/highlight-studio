"""Exercise the SSH framing with real HTTP, including binary bodies and Range."""
import io
import base64
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from scripts.ssh_http_relay import drain_response
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class DemoRelayTests(unittest.TestCase):
    def test_browser_abort_drains_response_before_next_request(self):
        payload = bytes(range(256)) * 1024
        frames = b''.join(base64.b64encode(payload[i:i+49152])+b'\n' for i in range(0,len(payload),49152))
        stream = io.BytesIO(frames+b'{"status":200}\n')
        class AbortedBrowser:
            def write(self, data): raise ConnectionAbortedError('Browser cancelled preview')
        drain_response(stream, AbortedBrowser(), len(payload))
        self.assertEqual(json.loads(stream.readline()), {'status':200})

    def test_binary_upload_download_and_multiple_requests(self):
        payload = bytes(range(256)) * 1024
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(201); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                data = payload[-16:] if self.headers.get('Range') else payload
                self.send_response(206 if self.headers.get('Range') else 200)
                self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
            def log_message(self,*args): pass
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            source = b''
            for method, body, headers in [('POST',payload,{}),('GET',b'',{'Range':'bytes=-16'}),('GET',b'',{})]:
                source += (json.dumps({'method':method,'path':'/test','length':len(body),'headers':headers})+'\n').encode()
                for i in range(0,len(body),49152):source+=base64.b64encode(body[i:i+49152])+b'\n'
            result = subprocess.run([sys.executable,str(ROOT/'scripts/ssh_http_relay.py'),'--port',str(server.server_port)],input=source,capture_output=True,check=True,timeout=20)
            stream=io.BytesIO(result.stdout); self.assertEqual(stream.readline(),b'HIGHLIGHT_RELAY_READY\n')
            for status,body in [(201,payload),(206,payload[-16:]),(200,payload)]:
                response=json.loads(stream.readline());self.assertEqual(response['status'],status)
                actual=b''
                while len(actual)<response['length']:actual+=base64.b64decode(stream.readline(),validate=False)
                self.assertEqual(actual,body)
            self.assertEqual(stream.read(),b'')
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_rejects_nonlocal_path(self):
        packet=json.dumps({'method':'GET','path':'http://example.com','length':0,'headers':{}}).encode()+b'\n'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ssh_http_relay.py'),'--port','8001'],input=packet,capture_output=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
