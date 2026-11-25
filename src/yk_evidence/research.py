"""Atomic, independently checkable integration of tabular research evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import ROOT, asset
from .review import read_reviews

MODELS = ('logistic_regression', 'random_forest', 'hist_gradient_boosting')
EVALUATION_FILES = ('evaluation_summary.json', 'evaluation_predictions.csv',
                    'evaluation_manifest.json', 'evaluation_stress.json', 'evaluation_capacity.json')
REQUIRED = EVALUATION_FILES + tuple(f'model_{name}.joblib' for name in MODELS) + (
    'run_metadata.json', 'source_metadata.json', 'source_profile.json',
    'corpus.json', 'questions.json', 'retrieval_evidence.json', 'release_gate.json', 'review_report.html')
MANIFEST = 'bundle_manifest.json'
ULB_SOURCE_URL = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'
ULB_SOURCE_SHA256 = '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'


def validate_source_identity(metadata: dict, profile: dict) -> None:
    """Bind the public benchmark label to the inspected publisher snapshot.

    This detects source substitution and false classification. A saved hash is
    not an independent signature, and offline verification without the raw CSV
    cannot prove the origin of adversarially rewritten prediction rows.
    """
    actual = profile.get('source_sha256')
    for key in ('csv_sha256', 'source_sha256'):
        claimed = metadata.get(key)
        if claimed is not None and (not isinstance(claimed, str) or claimed.lower() != actual):
            raise ValueError(f'source hash disagrees with metadata {key}')
    if metadata.get('classification') == 'public_ulb_creditcard':
        if metadata.get('source_url') != ULB_SOURCE_URL or actual != ULB_SOURCE_SHA256:
            raise ValueError('ULB public classification requires the pinned publisher URL and source SHA-256')
        if not any(metadata.get(key) for key in ('csv_sha256', 'source_sha256')):
            raise ValueError('ULB public metadata must record the pinned source SHA-256')
        expected = {'row_count': 284807, 'positive_count': 492, 'source_bytes': 150828752,
                    'time_range_seconds': [0.0, 172792.0]}
        if any(profile.get(key) != value for key, value in expected.items()):
            raise ValueError('ULB public source profile differs from the pinned snapshot')


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def code_identity() -> dict:
    fingerprint = hashlib.sha256()
    source_root = Path(__file__).parent
    for path in sorted(source_root.rglob('*.py')):
        fingerprint.update(path.relative_to(source_root).as_posix().encode())
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    revision = 'source-sha256:' + digest
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
        if Path(git('rev-parse', '--show-toplevel')).resolve() == ROOT.resolve() and not git('status', '--porcelain'):
            revision = git('rev-parse', 'HEAD')
    except (OSError, subprocess.SubprocessError):
        pass
    return {'code_sha256': digest, 'code_revision': revision}


def policy_for(classification: str) -> dict:
    return {
        'evidence_completeness': {'status': 'pass', 'scope': 'Required files, recorded arithmetic and citations verified; not independent source authentication.'},
        'research_suitability': {'status': 'historical_benchmark_only' if classification == 'public_ulb_creditcard' else 'smoke_only',
                                'scope': 'Historical anonymous tabular benchmark.' if classification == 'public_ulb_creditcard' else 'Generated fixture for software checks, not real fraud performance.'},
        'production_approval': {'status': 'blocked', 'reasons': ['no_institutional_pilot', 'no_production_authorization'],
                                'scope': 'Research evidence does not authorize operational deployment.'},
        'stress_interpretation': 'Failed stress criteria remain valid negative findings; inspect evaluation_stress.json.'}


def research_run(output_root: Path, run_id: str, csv_path: Path, metadata: dict,
                 corpus: dict, questions: list[str], seed: int = 42,
                 load_rows: int = 1_000_000, fpr_ceiling: float = .01) -> Path:
    """Finalize only verified output; the externally licensed raw CSV stays outside."""
    from .evaluation import evaluate_dataset
    from .retrieval import run_retrieval
    from .source_profile import profile_ulb
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('run_id must be 1-64 safe filename characters')
    if not isinstance(metadata, dict) or any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
                                           for k in ('classification', 'source_url', 'license_status', 'dataset_name')):
        raise ValueError('source metadata requires classification, source_url, license_status and dataset_name')
    if metadata['classification'] not in ('public_ulb_creditcard', 'synthetic_smoke_fixture'):
        raise ValueError('unsupported source classification')
    if not isinstance(questions, list) or not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError('questions must be a nonempty JSON list of question strings')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    if isinstance(load_rows, bool) or not isinstance(load_rows, int) or not 1 <= load_rows <= 10_000_000:
        raise ValueError('load_rows must be between 1 and 10000000')
    output_root = Path(output_root)
    if output_root.is_symlink():
        raise ValueError('output root may not be a symbolic link')
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / run_id
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(f'run already exists: {bundle}')
    lock = output_root / ('.' + run_id + '.lock')
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock_fd)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix='.' + run_id + '-', dir=output_root))
        profile = profile_ulb(Path(csv_path))
        validate_source_identity(metadata, profile)
        # Both acquisition helpers and the evaluator receive the same digest;
        # accept either historical field name, reject contradictions above.
        metadata = {**metadata, 'csv_sha256': profile['source_sha256'],
                    'source_sha256': profile['source_sha256']}
        retrieval = run_retrieval(corpus, questions)
        summary = evaluate_dataset(Path(csv_path), metadata, stage, seed=seed,
                                   fpr_ceiling=fpr_ceiling, load_rows=load_rows)
        if (summary['dataset']['source_sha256'] != profile['source_sha256']
                or summary['dataset']['row_count'] != profile['row_count']
                or sum(p['positive_count'] for p in summary['partitions'].values()) != profile['positive_count']):
            raise ValueError('source changed between profiling and evaluation')
        write(stage / 'source_metadata.json', metadata)
        write(stage / 'source_profile.json', profile)
        write(stage / 'corpus.json', corpus)
        write(stage / 'questions.json', questions)
        write(stage / 'retrieval_evidence.json', retrieval)
        version = asset('VERSION').read_text().strip()
        write(stage / 'run_metadata.json', {
            'schema_version': '0.4', 'run_id': run_id, 'project_version': version,
            'completed': True, 'utc_time': datetime.now(timezone.utc).isoformat(),
            'python_version': platform.python_version(), 'seed': seed,
            'source_classification': metadata['classification'],
            'source_sha256': profile['source_sha256'], **code_identity(),
            'raw_source_included': False,
            'distribution_note': 'Raw licensed CSV excluded. Predictions and model files are derived artifacts; source rights remain applicable.'})
        write(stage / 'release_gate.json', policy_for(metadata['classification']))
        from .research_view import render_report
        (stage / 'review_report.html').write_bytes(render_report(stage))
        write(stage / MANIFEST, {'schema_version': '0.4', 'run_id': run_id,
                                'project_version': version, 'source_classification': metadata['classification'],
                                'required_artifacts': list(REQUIRED),
                                'artifacts': {name: sha(stage / name) for name in REQUIRED}})
        errors = verify_research(stage, check_reviews=False)
        if errors:
            raise ValueError('research verification failed: ' + '; '.join(errors))
        if bundle.exists() or bundle.is_symlink():
            raise FileExistsError(f'run already exists: {bundle}')
        stage.rename(bundle)
        stage = None
        return bundle
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def research_smoke(output_root: Path, run_id: str, seed: int = 42, load_rows: int = 100_000) -> Path:
    """Fit actual models on clearly labelled generated rows, without network access."""
    with tempfile.TemporaryDirectory(prefix='yk-research-smoke-') as tmp:
        source = Path(tmp) / 'generated.csv'
        with source.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Time', *(f'V{i}' for i in range(1, 29)), 'Amount', 'Class'])
            for index in range(400):
                label = int(index % 8 == 0)
                values = [round((index % (7 + j % 5)) * .1 + label * (j % 3), 3) for j in range(1, 29)]
                writer.writerow([index // 2, *values, index % 100, label])
        metadata = {'dataset_name': 'Generated ULB-schema software smoke fixture',
                    'classification': 'synthetic_smoke_fixture', 'source_url': 'local:yk_evidence.research.research_smoke',
                    'license_status': 'project-generated synthetic fixture'}
        corpus = {'schema_version': '0.4', 'documents': [
            {'id': 'acme_revenue', 'title': 'Generated Acme revenue example', 'source_url': 'https://example.org/synthetic/acme',
             'classification': 'synthetic_smoke_fixture', 'text': 'Acme revenue was USD 100 in fiscal 2025. This is generated demonstration text.'},
            {'id': 'beta_cash', 'title': 'Generated Beta cash example', 'source_url': 'https://example.org/synthetic/beta',
             'classification': 'synthetic_smoke_fixture', 'text': 'Beta cash flow was USD 50 in fiscal 2025. This is generated demonstration text.'}]}
        return research_run(output_root, run_id, source, metadata, corpus,
                            ['Acme revenue fiscal 2025', 'unicorn zeppelin nebula'], seed, load_rows)


def verify_research(bundle: Path, check_reviews: bool = True) -> list[str]:
    """Check untrusted artifacts without loading executable model serialization."""
    from .evaluation import verify_evaluation
    from .retrieval import verify_retrieval
    bundle = Path(bundle)
    errors = []
    try:
        if not bundle.is_dir() or bundle.is_symlink() or (bundle / MANIFEST).is_symlink():
            return ['unsafe research bundle or manifest']
        manifest = read(bundle / MANIFEST)
        if not isinstance(manifest, dict) or manifest.get('schema_version') != '0.4':
            return ['unsupported research schema']
        artifacts = manifest.get('artifacts')
        if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED) or manifest.get('required_artifacts') != list(REQUIRED):
            return ['research artifact allowlist differs from required contract']
        if {p.name for p in bundle.iterdir()} != set(REQUIRED) | {MANIFEST}:
            return ['unlisted or missing research files']
        for name, fingerprint in artifacts.items():
            path = bundle / name
            if not re.fullmatch(r'[a-zA-Z0-9_.-]+', name) or not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
                errors.append('invalid artifact name or hash: ' + str(name))
            elif path.is_symlink() or not path.is_file():
                errors.append('unsafe research artifact: ' + name)
            elif sha(path) != fingerprint:
                errors.append('hash mismatch: ' + name)
        if errors:
            return errors
        errors.extend(verify_evaluation(bundle))
        corpus = read(bundle / 'corpus.json'); retrieval = read(bundle / 'retrieval_evidence.json')
        errors.extend(verify_retrieval(corpus, retrieval))
        questions = read(bundle / 'questions.json')
        if questions != [q['question'] for q in retrieval['queries']]:
            errors.append('archived questions disagree with retrieval')
        meta = read(bundle / 'run_metadata.json'); source = read(bundle / 'source_metadata.json')
        profile = read(bundle / 'source_profile.json'); summary = read(bundle / 'evaluation_summary.json')
        validate_source_identity(source, profile)
        if source != summary['dataset']['metadata']:
            errors.append('source metadata disagrees with evaluation')
        if source.get('classification') not in ('public_ulb_creditcard', 'synthetic_smoke_fixture'):
            errors.append('unsupported source classification')
        for key in ('source_classification', 'run_id', 'project_version'):
            if not meta.get(key) or meta[key] != manifest.get(key):
                errors.append('manifest/metadata mismatch: ' + key)
        if meta.get('source_classification') != source.get('classification'):
            errors.append('source classification mismatch')
        if (meta.get('schema_version') != '0.4' or meta.get('completed') is not True
                or meta.get('raw_source_included') is not False or meta.get('seed') != summary['config']['seed']):
            errors.append('invalid run completion/schema/seed/raw-source claim')
        if not re.fullmatch(r'[0-9a-f]{64}', str(meta.get('code_sha256', ''))) or not meta.get('code_revision'):
            errors.append('missing source code identity')
        if (profile.get('source_sha256') != summary['dataset']['source_sha256']
                or meta.get('source_sha256') != profile.get('source_sha256')
                or profile.get('row_count') != summary['dataset']['row_count']
                or profile.get('positive_count') != sum(p['positive_count'] for p in summary['partitions'].values())
                or profile.get('time_range_seconds') != [summary['partitions']['train']['time_min'], summary['partitions']['test']['time_max']]
                or profile.get('invalid_count') != 0):
            errors.append('source profile disagrees with evaluation')
        if read(bundle / 'release_gate.json') != policy_for(source['classification']):
            errors.append('unsupported research or production policy claim')
        from .research_view import render_report
        if (bundle / 'review_report.html').read_bytes() != render_report(bundle):
            errors.append('saved review report disagrees with evidence')
        if check_reviews:
            read_reviews(bundle)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        errors.append('invalid research bundle: ' + str(exc))
    return errors
