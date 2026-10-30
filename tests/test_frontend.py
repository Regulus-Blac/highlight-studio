"""Frontend asset boundary checks against the real local HTTP server."""
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from apps.backend.server import make_server

class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tempfile.TemporaryDirectory()
        cls.server = make_server(port=0, runtime=cls.runtime.name)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls.runtime.cleanup()

    def test_assets_have_browser_mime_types(self):
        for path, content_type in [('/', 'text/html'), ('/assets/app.js', 'text/javascript'), ('/assets/styles.css', 'text/css')]:
            with self.subTest(path=path), urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(content_type, response.headers['Content-Type'])
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                self.assertGreater(len(response.read()), 100)

    def test_assets_do_not_expose_repository_or_runtime(self):
        for path in ['/assets/../AGENTS.md', '/assets/%2e%2e/AGENTS.md', '/assets/server.py', '/runtime/tasks.sqlite3']:
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            self.assertEqual(error.exception.code, 404)

if __name__ == '__main__':
    unittest.main()
