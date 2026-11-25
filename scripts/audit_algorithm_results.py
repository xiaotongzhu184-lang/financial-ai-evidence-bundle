#!/usr/bin/env python3
"""Independently recompute seven ablations using only the Python standard library.

This checks saved predictions, not the origin of raw data or unpickled estimators.
Run: python scripts/audit_algorithm_results.py path/to/bundle [path/to/bundle ...]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(bundle):
    read = lambda name: json.loads((bundle / name).read_text())
    manifest = read('bundle_manifest.json')
    for name, expected in manifest['artifacts'].items():
        require(Path(name).name == name, 'unsafe artifact name')
        require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected, name)
    data, policy, results = map(read, ('algorithm_scores.json', 'fusion_policy.json', 'algorithm_results.json'))
    test, names = data['test'], policy['model_names']
    truth = [r['label'] for r in test]
    positives = sum(truth)
    output = []
    for model in results['ablations']:
        scores = [sum(r['models'][name]['calibrated_score'] * w
                      for name, w in zip(names, model['weights'])) for r in test]
        # AP integrates precision at each distinct score threshold; tied scores enter together.
        groups = {}
        for score, label in zip(scores, truth):
            totals = groups.setdefault(score, [0, 0])
            totals[0] += 1
            totals[1] += label
        seen = hits = 0
        ap = 0.0
        for score in sorted(groups, reverse=True):
            count, positive = groups[score]
            seen += count
            hits += positive
            ap += (positive / positives) * (hits / seen)
        eligible = [i for i, s in enumerate(scores) if s > model['threshold']]
        eligible.sort(key=lambda i: (-scores[i], test[i]['row_id']))
        capacity = math.floor(policy['budget_fraction'] * len(test))
        picked = set(eligible[:capacity])
        tp = sum(truth[i] for i in picked)
        fp = len(picked) - tp
        counts = {'rows': len(test), 'tp': tp, 'fp': fp, 'fn': positives - tp,
                  'tn': len(test) - positives - fp}
        brier = sum((s - y) ** 2 for s, y in zip(scores, truth)) / len(test)
        actual = model['test_metrics']
        require(counts == actual['counts'], model['name'] + ' confusion counts')
        require(math.isclose(ap, actual['average_precision'], abs_tol=1e-10), model['name'] + ' AP')
        require(math.isclose(brier, actual['brier'], abs_tol=1e-10), model['name'] + ' Brier')
        require(len(picked) == actual['reviewed_count'], model['name'] + ' reviewed')
        require(len(eligible) - len(picked) == actual['deferred_count'], model['name'] + ' deferred')
        output.append({'model': model['name'], 'average_precision': ap, 'brier': brier,
                       'counts': counts, 'reviewed': len(picked), 'status': 'pass'})
    return {'bundle': str(bundle), 'models': output, 'status': 'pass'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps({'status': 'pass', 'audits': [audit(p) for p in args.bundles]}, indent=2))
