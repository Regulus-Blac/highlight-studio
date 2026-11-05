"""Private framed HTTP relay over an authenticated SSH stdin/stdout channel."""
import argparse
import base64
import http.client
import json
import sys

HOP = {'connection', 'transfer-encoding', 'keep-alive', 'proxy-connection', 'upgrade'}

def receive_chunks(source, count):
    while count:
        data = base64.b64decode(source.readline(65538).strip(), validate=True)
        if not data or len(data) > min(count,49152):
            raise ValueError('Invalid transfer frame')
        count -= len(data)
        yield data

def send_chunks(source, target, count):
    while count:
        data = source.read(min(count,49152))
        if not data: raise EOFError('Response interrupted')
        target.write(base64.b64encode(data)+b'\n')
        count -= len(data)

def drain_response(source, target, count):
    """Keep framing aligned when a browser cancels a video Range request."""
    for data in receive_chunks(source, count):
        if target is not None:
            try:
                target.write(data)
            except OSError:
                target = None

def serve(port, source, target):
    target.write(b'HIGHLIGHT_RELAY_READY\n'); target.flush()
    while line := source.readline(65537):
        if len(line) > 65536:
            raise ValueError('Header too large')
        request = json.loads(line)
        path = request['path']
        if not path.startswith('/') or path.startswith('//') or '\r' in path or '\n' in path:
            raise ValueError('Only local relative paths are allowed')
        count = int(request['length'])
        if not 0 <= count <= 512 * 1024 * 1024:
            raise ValueError('Body exceeds upload limit')
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
        try:
            conn.putrequest(request['method'], path)
            for key, value in request['headers'].items():
                if key.lower() not in HOP | {'host', 'content-length'}:
                    conn.putheader(key, value)
            conn.putheader('Content-Length', str(count)); conn.endheaders()
            for chunk in receive_chunks(source, count):
                conn.send(chunk)
            response = conn.getresponse()
            headers = {k: v for k, v in response.getheaders() if k.lower() not in HOP}
            length = int(response.getheader('Content-Length', '0'))
            target.write((json.dumps({'status': response.status, 'headers': headers, 'length': length})+'\n').encode()); target.flush()
            send_chunks(response, target, length); target.flush()
        finally:
            conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535: parser.error('Invalid port')
    serve(args.port, sys.stdin.buffer, sys.stdout.buffer)
