"""Bind an unchanged transaction experiment and financial retrieval for one review.

The two sources share a review envelope, not a claim that a filing explains a
transaction. Existing model evidence can be reused without training it again.
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import os
from pathlib import Path
import re
import shutil
import tempfile

from .pipeline import asset
from .research import code_identity, read, sha, write
from .retrieval import run_retrieval, verify_retrieval
from .review import read_reviews

SCHEMA = '0.7-project'
MANIFEST = 'bundle_manifest.json'
EXECUTION = 'verified_component_assembly; training provenance retained in risk/run_metadata.json'
TOP_FILES = ('run_metadata.json', 'lineage.json', 'corpus.json', 'questions.json',
             'retrieval_evidence.json', 'release_gate.json', 'review_report.html')


def _lineage(bundle: Path, run_id: str) -> dict:
    risk = read(bundle / 'risk' / MANIFEST)
    return {'bundle_id': run_id, 'risk_run_id': risk['run_id'],
            'risk_schema': risk['schema_version'],
            'risk_manifest_sha256': sha(bundle / 'risk' / MANIFEST),
            'corpus_file_sha256': sha(bundle / 'corpus.json'),
            'questions_file_sha256': sha(bundle / 'questions.json'),
            'relationship': 'independent_sources_in_one_review; no_transaction_to_issuer_link'}


def _gate(bundle: Path) -> dict:
    return {**read(bundle / 'risk/release_gate.json'),
            'financial_analysis': {'status': 'extractive_only',
                                   'reason': 'Literal financial passages; no generative model or entailment check.'}}


def assemble_project(output_root: Path, run_id: str, risk_bundle: Path,
                     corpus: dict, questions: list[str]) -> Path:
    """Copy only verified evidence; never copy raw CSVs, journals or other files."""
    from .algorithm import verify_algorithm
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('run_id must be 1-64 safe filename characters')
    output_root, risk_bundle = Path(output_root), Path(risk_bundle)
    if output_root.is_symlink():
        raise ValueError('output root may not be a symbolic link')
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'run already exists: {destination}')
    errors = verify_algorithm(risk_bundle)
    if errors:
        raise ValueError('risk component verification failed: ' + '; '.join(errors))
    risk_manifest = read(risk_bundle / MANIFEST)
    if risk_manifest['schema_version'] != '0.6-algorithm':
        raise ValueError('project requires a schema 0.6 algorithm bundle')
    retrieval = run_retrieval(corpus, questions)
    lock = output_root / ('.' + run_id + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix='.' + run_id + '-', dir=output_root))
        (stage / 'risk').mkdir()
        for name in (*risk_manifest['artifacts'], MANIFEST):
            shutil.copyfile(risk_bundle / name, stage / 'risk' / name)
        write(stage / 'corpus.json', corpus)
        write(stage / 'questions.json', questions)
        write(stage / 'retrieval_evidence.json', retrieval)
        version = asset('VERSION').read_text().strip()
        write(stage / 'run_metadata.json', {'schema_version': SCHEMA, 'run_id': run_id,
              'project_version': version, 'completed': True,
              'utc_time': datetime.now(timezone.utc).isoformat(), **code_identity(),
              'execution': EXECUTION})
        write(stage / 'lineage.json', _lineage(stage, run_id))
        write(stage / 'release_gate.json', _gate(stage))
        (stage / 'review_report.html').write_bytes(render_project_report(stage))
        names = list(TOP_FILES) + ['risk/' + name for name in (*risk_manifest['artifacts'], MANIFEST)]
        write(stage / MANIFEST, {'schema_version': SCHEMA, 'run_id': run_id,
              'project_version': version, 'required_artifacts': sorted(names),
              'artifacts': {name: sha(stage / name) for name in sorted(names)}})
        errors = verify_project(stage, check_reviews=False)
        if errors:
            raise ValueError('project verification failed: ' + '; '.join(errors))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'run already exists: {destination}')
        stage.rename(destination)
        stage = None
        return destination
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def project_run(output_root: Path, run_id: str, csv_path: Path, metadata: dict,
                corpus: dict, questions: list[str], seed: int = 42,
                load_rows: int = 100_000, budget_fraction: float = .005) -> Path:
    from .algorithm import algorithm_run
    # Reject bad retrieval inputs before expensive model fitting.
    run_retrieval(corpus, questions)
    if (Path(output_root) / run_id).exists():
        raise FileExistsError(f'run already exists: {run_id}')
    with tempfile.TemporaryDirectory(prefix='yk-project-') as tmp:
        risk = algorithm_run(Path(tmp), run_id, csv_path, metadata, seed, load_rows, budget_fraction)
        return assemble_project(output_root, run_id, risk, corpus, questions)


def project_smoke(output_root: Path, run_id: str, seed: int = 42, load_rows: int = 100_000) -> Path:
    from .algorithm import algorithm_smoke
    corpus = {'schema_version': '0.4', 'documents': [
        {'id': 'generated_revenue', 'title': 'Generated revenue fixture',
         'source_url': 'https://example.org/synthetic/revenue',
         'classification': 'synthetic_smoke_fixture',
         'text': 'Acme revenue was USD 100 in fiscal 2025. Generated software test text.'}]}
    if (Path(output_root) / run_id).exists():
        raise FileExistsError(f'run already exists: {run_id}')
    with tempfile.TemporaryDirectory(prefix='yk-project-smoke-') as tmp:
        risk = algorithm_smoke(Path(tmp), run_id, seed, load_rows)
        return assemble_project(output_root, run_id, risk, corpus,
                                ['Acme revenue fiscal 2025', 'unicorn nebula'])


def verify_project(bundle: Path, check_reviews: bool = True) -> list[str]:
    """Validate the envelope and both components without loading saved models."""
    from .algorithm import REQUIRED_RESILIENCE, verify_algorithm
    bundle = Path(bundle)
    errors = []
    try:
        if not bundle.is_dir() or bundle.is_symlink():
            return ['unsafe project bundle']
        if any(path.is_symlink() for path in bundle.rglob('*')):
            return ['project bundle may not contain symbolic links']
        manifest = read(bundle / MANIFEST)
        expected = set(TOP_FILES) | {'risk/' + name for name in (*REQUIRED_RESILIENCE, MANIFEST)}
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA
                or set(manifest.get('artifacts', {})) != expected
                or manifest.get('required_artifacts') != sorted(expected)):
            return ['invalid project manifest or component inventory']
        actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
        if actual != expected | {MANIFEST}:
            return ['project file inventory mismatch']
        for name in sorted(expected):
            if sha(bundle / name) != manifest['artifacts'][name]:
                errors.append('hash mismatch: ' + name)
        if errors:
            return errors
        errors.extend('risk: ' + error for error in verify_algorithm(bundle / 'risk', check_reviews=False))
        evidence = read(bundle / 'retrieval_evidence.json')
        errors.extend(verify_retrieval(read(bundle / 'corpus.json'), evidence))
        if [query['question'] for query in evidence['queries']] != read(bundle / 'questions.json'):
            errors.append('retrieval questions differ from archived input')
        metadata = read(bundle / 'run_metadata.json')
        if (metadata.get('schema_version') != SCHEMA or metadata.get('completed') is not True
                or metadata.get('run_id') != manifest.get('run_id')
                or metadata.get('project_version') != manifest.get('project_version')
                or metadata.get('execution') != EXECUTION
                or not re.fullmatch(r'[0-9a-f]{64}', str(metadata.get('code_sha256', '')))
                or not isinstance(metadata.get('code_revision'), str) or not metadata['code_revision'].strip()
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', metadata.get('run_id', ''))):
            errors.append('project metadata mismatch')
        if read(bundle / 'lineage.json') != _lineage(bundle, manifest['run_id']):
            errors.append('component lineage mismatch')
        if read(bundle / 'release_gate.json') != _gate(bundle):
            errors.append('project gate differs from component evidence')
        if (bundle / 'review_report.html').read_bytes() != render_project_report(bundle):
            errors.append('project report differs from evidence')
        if check_reviews:
            read_reviews(bundle)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append('invalid project evidence: ' + str(exc))
    return errors


def render_project_report(bundle: Path, csrf_token: str | None = None) -> bytes:
    esc = lambda value: html.escape(str(value), quote=True)
    metadata = read(bundle / 'run_metadata.json')
    lineage = read(bundle / 'lineage.json')
    results = read(bundle / 'risk/algorithm_results.json')
    retrieval = read(bundle / 'retrieval_evidence.json')
    gate = read(bundle / 'release_gate.json')
    source = read(bundle / 'risk/source_metadata.json')
    documents = ''.join('<li><code>' + esc(doc['id']) + '</code>: ' +
                       esc(doc['classification']) + ' · <a rel="noreferrer" href="' +
                       esc(doc['source_url']) + '">Source</a></li>'
                       for doc in read(bundle / 'corpus.json')['documents'])
    comparisons = ''.join('<tr><th>' + esc(item['name']) + '</th><td>' +
        format(item['test_metrics']['average_precision'], '.6f') + '</td><td>' +
        esc(item['test_metrics']['reviewed_count']) + '</td></tr>' for item in results['ablations'])
    passages = ''.join('<article><h3>' + esc(q['question']) + '</h3><p>' + esc(q['status']) +
        '</p><blockquote>' + esc(q['answer_excerpt'] or q['reason']) + '</blockquote>' +
        ('<a rel="noreferrer" href="' + esc(q['citation']['source_url']) + '">' +
         esc(q['citation']['document_id']) + '</a>' if q['citation'] else '') + '</article>'
        for q in retrieval['queries'])
    review = ''
    if csrf_token is not None:
        history = ''.join('<li>' + esc(item['reviewer']) + ': ' + esc(item['decision']) +
                          ' — ' + esc(item['rationale']) + '</li>' for item in read_reviews(bundle))
        review = '<section><h2>Record evidence review</h2><p>This does not approve production use.</p>' + \
            '<form method="post" action="/api/reviews"><input type="hidden" name="csrf" value="' + esc(csrf_token) + \
            '"><label>Reviewer<input name="reviewer" required minlength="2" maxlength="80"></label>' + \
            '<label>Decision<select name="decision"><option value="request_changes">Request changes</option>' + \
            '<option value="accept_demo_evidence">Accept demonstration evidence</option>' + \
            '<option value="reject_evidence">Reject evidence</option></select></label>' + \
            '<label>Reason<textarea name="rationale" required minlength="10" maxlength="2000"></textarea></label>' + \
            '<button>Save review</button></form><ul>' + history + '</ul></section>'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8">' + \
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Project evidence review</title>' + \
        '<style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.6 system-ui;color:#193c4a;background:#f1f5f6}' + \
        'section{background:white;margin:1rem 0;padding:1.5rem;border:1px solid #cbdadc}td,th{text-align:left;padding:.5rem}' + \
        'table{width:100%}blockquote{margin-left:0;border-left:3px solid #28746e;padding-left:1rem}' + \
        'code{overflow-wrap:anywhere}label{display:block;margin:.8rem 0}input,textarea,select{display:block;max-width:100%;font:inherit}' + \
        'a{color:#075d86}button{padding:.6rem;font:inherit}h1{line-height:1.2}</style></head><body>' + \
        '<h1>Financial research and transaction evidence</h1><p>Review bundle <code>' + esc(metadata['run_id']) + \
        '</code> · ' + esc(metadata['project_version']) + '</p><nav><a href="/api/export">Download evidence ZIP</a> · ' + \
        '<a href="/api/manifest">Manifest</a> · <a href="/api/reviews">Review journal</a></nav>' + \
        '<section><h2>What this run establishes</h2><p>Financial passages and a transaction experiment share one verified review bundle. ' + \
        'They are independent sources: no issuer-to-transaction relationship is inferred.</p><p>Production approval: <strong>' + \
        esc(gate['production_approval']['status']) + '</strong>. Financial mode: extractive only, no generated answers.</p>' + \
        '<p>Transaction source: ' + esc(source['dataset_name']) + ' · <strong>' + esc(source['classification']) + \
        '</strong>.</p><p>Training run <code>' + esc(lineage['risk_run_id']) + '</code>; source evidence SHA-256 <code>' + \
        esc(lineage['risk_manifest_sha256']) + '</code>. Assembly preserves the original training provenance.</p></section>' + \
        '<section><h2>Frozen transaction models</h2><p>Average precision measures ranking on the recorded held-out data; ' + \
        'it does not prove current institutional performance. Generated fixtures validate software only.</p>' + \
        '<table><tr><th>Comparison</th><th>Test average precision</th><th>Reviewed rows</th></tr>' + comparisons + \
        '</table><p><a href="/api/artifacts/risk/algorithm_results.json">Raw comparison, stress and recovery results</a> · ' + \
        '<a href="/api/artifacts/risk/resilience_evidence.json">Recovery receipts</a></p></section>' + \
        '<section><h2>Source-linked financial passages</h2><p>Classification is declared per document. ' + \
        'synthetic_smoke_fixture means generated software test text.</p><ul>' + documents + '</ul>' + passages + '</section>' + review + \
        '<footer>Hashes establish internal consistency, not source authenticity or reviewer identity.</footer></body></html>'
    return document.encode('utf-8')
