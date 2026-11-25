#!/usr/bin/env python3
"""Exercise the installed project CLI, trusted batch scoring and real HTTP review."""
import argparse
import csv
import http.client
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import urlencode
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'python': sys.version, 'scope': 'generated software acceptance only', 'commands': []}
    def command(name, arguments, expected=0):
        completed = subprocess.run([sys.executable, '-m', 'yk_evidence', *arguments],
                                   capture_output=True, text=True, timeout=180)
        (root / (name + '.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
        result['commands'].append({'name': name, 'arguments': arguments, 'returncode': completed.returncode})
        require(completed.returncode == expected, f'{name} returned {completed.returncode}, expected {expected}')
    try:
        command('project', ['project-smoke', '--output', str(root), '--run-id', 'project', '--load-rows', '10000'])
        bundle = root / 'project'
        command('verify', ['verify', str(bundle)])
        source = root / 'unlabeled-generated.csv'
        with source.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_id', 'Time', *(f'V{i}' for i in range(1, 29)), 'Amount'])
            for index in range(240):
                pattern = int(index % 8 == 0)
                writer.writerow([1000 + index, index // 2,
                    *[round((index % (7 + j % 5)) * .1 + pattern * (j % 3), 3) for j in range(1, 29)], index % 100])
        scoring = ['score-batch', '--model-bundle', str(bundle / 'risk'), '--csv', str(source),
                   '--output', str(root), '--run-id', 'scores']
        command('untrusted-rejected', scoring, 2)
        require(not (root / 'scores').exists(), 'untrusted attempt created completed output')
        command('scoring', scoring + ['--trust-models', '--same-feature-space'])
        command('verify-scoring', ['verify', str(root / 'scores')])
        from yk_evidence.demo import build_server
        server = build_server(bundle, port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=90)
            connection.request('GET', '/')
            response = connection.getresponse(); page = response.read().decode()
            require(response.status == 200 and 'Frozen transaction models' in page and
                    'Source-linked financial passages' in page, 'both components not shown')
            (root / 'served-page.html').write_text(page, encoding='utf-8')
            token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
            body = urlencode({'csrf': token, 'reviewer': 'Automated acceptance', 'decision': 'request_changes',
                              'rationale': 'Software acceptance completed; human domain review is still required.'})
            connection.request('POST', '/api/reviews', body,
                {'Content-Type': 'application/x-www-form-urlencoded', 'Origin': f'http://127.0.0.1:{server.server_port}'})
            response = connection.getresponse(); response.read()
            require(response.status == 303, 'review was not saved')
            connection.request('GET', '/api/export')
            response = connection.getresponse(); exported = response.read()
            require(response.status == 200, 'export failed')
            with zipfile.ZipFile(io.BytesIO(exported)) as archive:
                manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
                require(set(archive.namelist()) == set(manifest['artifacts']) | {'bundle_manifest.json'}, 'ZIP inventory mismatch')
                require(archive.testzip() is None, 'ZIP CRC failed')
            (root / 'downloaded-evidence.zip').write_bytes(exported)
            connection.request('GET', '/', headers={'Host': 'unrelated.example'})
            response = connection.getresponse(); response.read()
            require(response.status == 403, 'foreign host was accepted')
            connection.close()
        finally:
            server.shutdown(); server.server_close(); worker.join()
        command('review-chain', ['verify', str(bundle)])
        broken = root / 'tampered'
        shutil.copytree(bundle, broken)
        (broken / 'risk/source_profile.json').write_text('{}', encoding='utf-8')
        command('tamper-rejected', ['verify', str(broken)], 1)
        result['status'] = 'pass'
        result['checks'] = ['fitted project', 'joint semantic verification', 'trust refusal', 'unlabeled batch scoring',
                            'HTTP page', 'review POST', 'nested ZIP export', 'foreign Host refusal', 'journal verification', 'tamper refusal']
    except Exception as exc:
        result['status'] = 'fail'; result['error'] = str(exc)
    (root / 'acceptance_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
