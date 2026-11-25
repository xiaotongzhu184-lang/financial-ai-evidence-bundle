"""Local source-aware review UI and verified, allowlisted evidence export."""
from __future__ import annotations

import html
import io
import json
import secrets
import zipfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .review import MAX_REVIEW_FORM_BYTES, append_review, read_reviews


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _json(value) -> str:
    return _escape(json.dumps(value, indent=2, ensure_ascii=False))


def _number(value) -> str:
    if isinstance(value, int):
        return str(value)
    return f'{value:.5f}' if isinstance(value, (int, float)) else _escape(value)


def render_report(bundle: Path, csrf_token: str | None = None) -> bytes:
    from .research import read
    metadata = read(bundle / 'run_metadata.json')
    source = read(bundle / 'source_metadata.json')
    profile = read(bundle / 'source_profile.json')
    summary = read(bundle / 'evaluation_summary.json')
    gate = read(bundle / 'release_gate.json')
    retrieval = read(bundle / 'retrieval_evidence.json')
    stress = read(bundle / 'evaluation_stress.json')
    stress = {**stress, 'scenarios': {
        name: {**{key: value for key, value in result.items() if key != 'predictions'},
               'saved_prediction_count': len(result.get('predictions', []))}
        for name, result in stress.get('scenarios', {}).items()}}
    capacity = read(bundle / 'evaluation_capacity.json')
    rows = []
    for name, model in summary['models'].items():
        validation = model['metrics']['validation']; test = model['metrics']['test']
        workload = test.get('review_workload', {})
        columns = [name, validation.get('average_precision'), test.get('average_precision'),
                   test.get('precision'), test.get('recall'), test.get('fpr'), test.get('brier'), model.get('threshold'),
                   workload.get('alert_count', 'Not recorded'), workload.get('alerts_per_10000', 'Not recorded'),
                   workload.get('missed_cases', 'Not recorded')]
        rows.append('<tr>' + ''.join('<td>' + _number(value) + '</td>' for value in columns) + '</tr>')
    citations = []
    for query in retrieval['queries']:
        citation = query.get('citation')
        reference = (f'<a href="{_escape(citation["source_url"])}" rel="noreferrer">{_escape(citation["document_id"])}</a>'
                     if citation else 'No supported passage')
        citations.append(f'<article><h3>{_escape(query["question"])}</h3><p class="tag">{_escape(query["status"])}</p>'
                         f'<blockquote>{_escape(query.get("answer_excerpt") or query.get("reason", "No evidence"))}</blockquote>'
                         f'<p>{reference}</p></article>')
    review = ''
    links = ''
    if csrf_token is not None:
        history = ''.join(f'<li><strong>{_escape(item["decision"])}</strong> — {_escape(item["reviewer"])}'
                          f'<p>{_escape(item["rationale"])}</p></li>' for item in read_reviews(bundle))
        review = f'''<section><h2>Record an evidence review</h2><p>This local name is self-declared. The hash chain does not establish reviewer identity or grant deployment authority.</p>
<form method="post" action="/api/reviews"><input type="hidden" name="csrf" value="{csrf_token}">
<label>Reviewer name<input name="reviewer" minlength="2" maxlength="80" required></label>
<label>Decision<select name="decision"><option value="request_changes">Request changes</option><option value="accept_demo_evidence">Accept demonstration evidence</option><option value="reject_evidence">Reject evidence</option></select></label>
<label>Reason<textarea name="rationale" minlength="10" maxlength="2000" rows="4" required></textarea></label>
<button type="submit">Save review</button></form><h3>Review history</h3><ol>{history or '<li>No review recorded.</li>'}</ol></section>'''
        links = '<nav><a href="/api/export">Download verified evidence ZIP</a> · <a href="/api/manifest">Manifest</a> · <a href="/api/reviews">Review journal</a></nav>'
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research evidence review — {_escape(metadata['run_id'])}</title><style>
body{{margin:0;background:#eef2f5;color:#203447;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:24px}}header{{background:#143448;color:#fff;padding:20px 0;border-bottom:5px solid #27796b}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.15}}h2{{font-size:23px}}h3{{font-size:18px}}section{{background:white;border:1px solid #d4dfe5;border-radius:8px;padding:24px;margin:20px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}.status{{font-size:22px;font-weight:700}}.blocked{{color:#a03b22}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d4dfe5;white-space:nowrap}}th{{background:#f0f5f7}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap;word-break:break-word}}pre{{font-size:13px;background:#f4f7f8;padding:14px}}a{{color:#075b89}}nav{{margin:20px 0}}blockquote{{margin-left:0;border-left:4px solid #27796b;padding-left:16px;overflow-wrap:anywhere}}article{{border-top:1px solid #d4dfe5;padding:12px 0}}form{{display:grid;gap:14px;max-width:650px}}label{{display:grid;gap:5px;font-weight:600}}input,select,textarea{{padding:10px;font:inherit;border:1px solid #91a8b5;border-radius:3px}}button{{padding:10px 18px;background:#1a665d;border:0;border-radius:3px;color:white;font:inherit;justify-self:start;cursor:pointer}}:focus-visible{{outline:3px solid #c88721;outline-offset:3px}}.tag{{font-size:14px;color:#526775}}summary{{cursor:pointer;font-weight:600}}footer{{color:#526775;font-size:14px}}@media(max-width:600px){{main{{padding:14px}}section{{padding:16px}}}}</style></head><body>
<header><main><p>Financial AI · Local research review</p><h1>Evidence you can inspect</h1><p>Run <code>{_escape(metadata['run_id'])}</code> · Version {_escape(metadata['project_version'])} · Schema 0.4</p></main></header><main>{links}
<div class="grid"><section><h2>Evidence completeness</h2><p class="status">{_escape(gate['evidence_completeness']['status'])}</p><p>Recorded files, supported calculations and citations. Re-run the verifier before relying on a downloaded report.</p></section>
<section><h2>Research suitability</h2><p class="status">{_escape(gate['research_suitability']['status'])}</p><p>{_escape(gate['research_suitability']['scope'])}</p></section>
<section><h2>Production approval</h2><p class="status blocked">{_escape(gate['production_approval']['status'])}</p><p>{_escape(gate['production_approval']['scope'])}</p></section></div>
<section><h2>Source and boundaries</h2><p><strong>{_escape(source['dataset_name'])}</strong> · {_escape(source['classification'])}</p><p>{profile['row_count']:,} rows. Declared rights: {_escape(source['license_status'])}.</p><p>Source: <code>{_escape(source['source_url'])}</code><br>SHA-256: <code>{_escape(profile['source_sha256'])}</code></p><p>The raw transaction CSV is excluded from this bundle. Saved scores, labels and model files are derived artifacts and retain source-related distribution considerations. Relative time has no invented calendar date. No customer-network interpretation is supported by this table.</p><details><summary>Chronological partitions</summary><pre>{_json(summary['partitions'])}</pre></details></section>
<section><h2>Models and operating thresholds</h2><p>Model choice uses validation average precision. Threshold choice uses the validation period and declared false-positive ceiling. The test period is held out.</p><div class="table"><table><thead><tr><th>Model</th><th>Validation AP</th><th>Test AP</th><th>Test precision</th><th>Test recall</th><th>Test FPR</th><th>Test Brier</th><th>Threshold</th><th>Alert count</th><th>Alerts per 10,000</th><th>Missed cases</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>Alert and missed-case counts are derived from the held-out confusion matrix. Alerts per 10,000 is a scaled observed rate, not a separate workload experiment. Review minutes and monetary costs were not measured. No institution-specific handling time was supplied.</p><details><summary>Selection, metrics and uncertainty</summary><pre>{_json({'selection': summary.get('selection'), 'models': summary['models']})}</pre></details></section>
<section><h2>Frozen-model stress</h2><p>A failed stress criterion is a negative experimental finding. It does not make a correctly recorded run disappear.</p><pre>{_json(stress)}</pre></section>
<section><h2>Measured scoring capacity</h2><p>Repeated batches measure processing load. They do not increase the number of independent accuracy observations. Memory and time figures depend on the recorded machine and run.</p><pre>{_json(capacity)}</pre></section>
<section><h2>Source-linked financial retrieval</h2><p>Method: {_escape(retrieval['method'])}. Literal extracts are retrieved from the archived corpus; no generative model or financial advice is produced.</p>{''.join(citations)}</section>{review}
<footer><p>Code: <code>{_escape(metadata['code_revision'])}</code></p><p>Hashes show internal consistency; they do not independently authenticate data origin, source rights or reviewer identity.</p></footer></main></body></html>'''
    return document.encode('utf-8')


def build_research_server(bundle: Path, host: str = '127.0.0.1', port: int = 8765):
    from .demo import LocalHTTPServer
    from .research import MANIFEST, REQUIRED, verify_research
    renderer = render_report
    verifier = verify_research
    candidate = json.loads((bundle / MANIFEST).read_text(encoding='utf-8'))
    if isinstance(candidate, dict) and candidate.get('schema_version') == '0.5-algorithm':
        from .algorithm import REQUIRED, verify_algorithm, render_algorithm_report
        renderer = render_algorithm_report
        verifier = verify_algorithm
    elif isinstance(candidate, dict) and candidate.get('schema_version') == '0.6-algorithm':
        from .algorithm import REQUIRED_RESILIENCE, verify_algorithm, render_resilience_report
        REQUIRED = REQUIRED_RESILIENCE
        renderer = render_resilience_report
        verifier = verify_algorithm
    elif isinstance(candidate, dict) and candidate.get('schema_version') == '0.7-project':
        from .project import verify_project, render_project_report
        REQUIRED = tuple(candidate.get('artifacts', {}))
        renderer = render_project_report
        verifier = verify_project
    if host != '127.0.0.1':
        raise ValueError('research viewer binds to IPv4 loopback only')
    errors = verifier(bundle)
    if errors:
        raise ValueError('bundle verification failed: ' + '; '.join(errors))
    token = secrets.token_urlsafe(32)
    allowed = set(REQUIRED) | {MANIFEST}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def verified(self):
            errors = verifier(bundle)
            if errors:
                self.send_error(409, 'bundle or journal verification failed')
                return False
            return True

        def send_payload(self, payload, content_type, attachment=None):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if not self.valid_host():
                self.send_error(403, 'invalid local host'); return
            if not self.verified():
                return
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                self.send_payload(renderer(bundle, token), 'text/html; charset=utf-8')
            elif path == '/api/manifest':
                self.send_payload((bundle / MANIFEST).read_bytes(), 'application/json')
            elif path == '/api/reviews':
                self.send_payload(json.dumps(read_reviews(bundle), indent=2).encode(), 'application/json')
            elif path == '/api/export':
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for name in sorted(allowed):
                        archive.write(bundle / name, arcname=name)
                self.send_payload(buffer.getvalue(), 'application/zip', 'research-evidence.zip')
            elif path.startswith('/api/artifacts/') and path.removeprefix('/api/artifacts/') in allowed:
                name = path.removeprefix('/api/artifacts/')
                self.send_payload((bundle / name).read_bytes(), 'application/octet-stream', Path(name).name)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != '/api/reviews':
                self.send_error(404); return
            if not self.valid_host():
                self.send_error(403, 'invalid local host'); return
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
                self.send_error(403, 'invalid origin'); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size > MAX_REVIEW_FORM_BYTES:
                    self.send_error(413, f'review form exceeds {MAX_REVIEW_FORM_BYTES}-byte transport limit')
                    return
                if size <= 0 or self.headers.get('Content-Type', '').split(';')[0] != 'application/x-www-form-urlencoded':
                    raise ValueError('invalid form size or media type')
                fields = parse_qs(self.rfile.read(size).decode('utf-8'), keep_blank_values=True)
                if set(fields) != {'csrf', 'reviewer', 'decision', 'rationale'} or any(len(v) != 1 for v in fields.values()) or fields['csrf'] != [token]:
                    raise ValueError('invalid review form token or fields')
                if not self.verified():
                    return
                append_review(bundle, fields['reviewer'][0], fields['decision'][0], fields['rationale'][0])
            except (OSError, ValueError, UnicodeError) as exc:
                self.send_error(400, str(exc)); return
            self.send_response(303); self.send_header('Location', '/')
            self.send_header('Content-Length', '0'); self.end_headers()

        def log_message(self, format, *args):
            pass

    return LocalHTTPServer((host, port), Handler)
