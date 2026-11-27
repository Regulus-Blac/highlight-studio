"""Run the actual CI entry point without network, Git mutations or real keys."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]


class CIHostKeyTests(unittest.TestCase):
    def run_entry(self, directory, meta, *, token='fixture-actions-token', key='fixture-not-a-real-private-key'):
        env = {'RUNNER_TEMP': directory, 'GITHUB_TOKEN': token, 'COMPETITION_READ_KEY': key}
        with patch.dict(os.environ, env), patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]), \
                patch('model_ref.git') as git, patch('subprocess.run') as command, \
                patch('urllib.request.urlopen', side_effect=meta) as fetch:
            output = io.StringIO()
            with redirect_stdout(output):
                runpy.run_path(str(ROOT/'scripts/ci_submodule.py'), run_name='__main__')
            return fetch, command, git, output.getvalue()

    def test_authenticated_meta_request_avoids_anonymous_rate_limit(self):
        def meta(request, timeout):
            # Reproduce the failing Actions boundary: anonymous /meta is exhausted.
            if request.get_header('Authorization') != 'Bearer fixture-actions-token':
                raise HTTPError(request.full_url, 403, 'rate limit exceeded', {}, None)
            self.assertEqual(request.full_url, 'https://api.github.com/meta')
            self.assertEqual(timeout, 30)
            response = io.StringIO(json.dumps({'ssh_keys': ['ssh-ed25519 fixture-host-key']}))
            return response

        with tempfile.TemporaryDirectory() as directory:
            fetch, command, git, output = self.run_entry(directory, meta)
            fetch.assert_called_once()
            command.assert_called_once()
            self.assertEqual(command.call_args.args[0][-4:], ['submodule', 'update', '--init', '--recursive'])
            ssh = command.call_args.kwargs['env']['GIT_SSH_COMMAND']
            self.assertIn('StrictHostKeyChecking=yes', ssh)
            self.assertIn('IdentitiesOnly=yes', ssh)
            self.assertEqual((Path(directory)/'competition-ssh/known_hosts').read_text(), 'github.com ssh-ed25519 fixture-host-key\n')
            self.assertEqual(git.call_count, 2)
            self.assertNotIn('fixture-actions-token', output)
            self.assertNotIn('fixture-not-a-real-private-key', output)

    def test_missing_actions_token_stops_before_anonymous_request(self):
        def meta(*args, **kwargs):
            self.fail('The CI entry must not fall back to unauthenticated API requests')
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(SystemExit, 'GITHUB_TOKEN'):
            self.run_entry(directory, meta, token='')

    def test_private_repository_key_is_still_required(self):
        def meta(*args, **kwargs):
            self.fail('Missing private read access must fail before contacting GitHub')
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(SystemExit, 'COMPETITION_READ_KEY'):
            self.run_entry(directory, meta, key='')


if __name__ == '__main__':
    unittest.main()
