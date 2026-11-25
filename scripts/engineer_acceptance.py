#!/usr/bin/env python3
"""Account-free engineer acceptance; keep each invocation in a new directory."""
import argparse
import http.client
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from urllib.parse import urlencode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('research', 'algorithm'), default='research')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = []
    def command(label, arguments, expected=0):
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name('run_local.py')),
                                    *map(str, arguments)], capture_output=True, text=True)
        (root / (label + '.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
        if completed.returncode != expected:
            raise RuntimeError(f'{label}: exit {completed.returncode}, expected {expected}; see its log')
        results.append({'check': label, 'status': 'pass', 'exit_code': completed.returncode})
    try:
        command('smoke', [args.mode + '-smoke', '--output', root, '--run-id', 'acceptance', '--load-rows', '100000'])
        bundle = root / 'acceptance'
        command('verify', ['verify', bundle])
        if args.mode == 'algorithm':
            manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
            if manifest['schema_version'] != '0.6-algorithm' or not {
                    'resilience_policy.json', 'resilience_evidence.json'}.issubset(manifest['artifacts']):
                raise RuntimeError('0.6 recovery evidence is missing')
            independent = subprocess.run([sys.executable, str(Path(__file__).with_name('audit_resilience_results.py')),
                                          str(bundle)], capture_output=True, text=True)
            (root / 'independent-resilience-audit.log').write_text(independent.stdout + independent.stderr,
                                                                    encoding='utf-8')
            if independent.returncode:
                raise RuntimeError('Independent resilience audit failed')
            results.append({'check': 'independent-resilience-audit', 'status': 'pass'})
        tampered = root / 'tampered'
        shutil.copytree(bundle, tampered)
        (tampered / 'source_profile.json').write_text('{}', encoding='utf-8')
        command('tamper-rejected', ['verify', tampered], expected=1)
        from yk_evidence.demo import build_server
        server = build_server(bundle, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=30)
            client.request('GET', '/')
            response = client.getresponse(); page = response.read().decode()
            if response.status != 200:
                raise RuntimeError('Review page not available')
            if args.mode == 'algorithm':
                if ('Prediction delivery recovery' not in page or
                        page.index('Prediction delivery recovery') >
                        page.index('Unprotected 0.5 control probes (before recovery)')):
                    raise RuntimeError('0.6 recovery UI is missing or behind the unprotected controls')
                for name in ('resilience_policy.json', 'resilience_evidence.json'):
                    client.request('GET', '/api/artifacts/' + name)
                    response = client.getresponse()
                    payload = response.read()
                    if response.status != 200 or payload != (bundle / name).read_bytes():
                        raise RuntimeError('Recovery artifact HTTP download differs: ' + name)
            token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
            body = urlencode({'csrf': token, 'reviewer': 'Automated acceptance check', 'decision': 'request_changes',
                              'rationale': 'Software round trip completed; human engineering acceptance is still required.'})
            client.request('POST', '/api/reviews', body=body, headers={'Content-Type': 'application/x-www-form-urlencoded',
                           'Origin': f'http://127.0.0.1:{server.server_port}'})
            response = client.getresponse(); response.read()
            if response.status != 303:
                raise RuntimeError('Review record was not saved')
            client.request('GET', '/api/export')
            response = client.getresponse(); exported = response.read()
            if response.status != 200:
                raise RuntimeError('Evidence export failed')
            if args.mode == 'algorithm':
                with zipfile.ZipFile(io.BytesIO(exported)) as archive:
                    expected = set(manifest['artifacts']) | {'bundle_manifest.json'}
                    if set(archive.namelist()) != expected or archive.testzip() is not None:
                        raise RuntimeError('0.6 recovery export content differs')
            (root / 'reviewed-evidence.zip').write_bytes(exported)
            client.close()
            results.append({'check': 'http-review-and-export', 'status': 'pass'})
        finally:
            server.shutdown(); server.server_close(); thread.join()
        command('review-chain-verify', ['verify', bundle])
        result = {'status': 'pass', 'checks': results, 'scope': 'Generated offline software acceptance; not a real-data or human review result.'}
    except Exception as exc:
        result = {'status': 'fail', 'checks': results, 'error': str(exc)}
    (root / 'acceptance_result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
