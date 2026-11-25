"""Bounded score-delivery recovery with validation-frozen fallback policies.

Local receipts detect accidental corruption after model production. They are
not independent signatures and cannot detect plausible pre-receipt errors.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Set
from itertools import combinations

import numpy as np
from sklearn.metrics import average_precision_score
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_none

from .fusion import (_binary_labels, _fraction, _probabilities, _row_ids,
                     _split_after_ties, _time_values, apply_policy)


def fit_resilience_policy(validation_scores, labels, times, row_ids, model_names,
                          budget_fraction):
    """Choose each subset's model on validation first half; freeze on second."""
    if isinstance(model_names, (str, bytes, Mapping, Set)):
        raise ValueError('four distinct model names required')
    try:
        names = list(model_names)
    except TypeError as error:
        raise ValueError('four distinct model names required') from error
    if (len(names) != 4 or any(type(name) is not str or not name.strip() for name in names)
            or len(set(names)) != 4):
        raise ValueError('four distinct model names required')
    rows = [_probabilities(row) for row in validation_scores]
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError('validation_scores must be nonempty n-by-4 probabilities')
    matrix = np.asarray(rows)
    ids = _row_ids(row_ids, len(rows))
    truth = _binary_labels(labels, len(rows))
    clock = _time_values(times, len(rows))
    budget = _fraction(budget_fraction)
    tuning, freeze = _split_after_ties(sorted(range(len(rows)), key=lambda i: (clock[i], ids[i])), clock)
    if {truth[i] for i in tuning} != {0, 1}:
        raise ValueError('validation tuning period must include both classes')
    capacity = math.floor(budget * len(tuning))
    metrics = {}
    for j, name in enumerate(names):
        top = sorted(tuning, key=lambda i: (-matrix[i,j], ids[i]))[:capacity]
        metrics[name] = {'average_precision': float(average_precision_score([truth[i] for i in tuning], matrix[tuning,j])),
                         'budget_recall': sum(truth[i] for i in top) / sum(truth[i] for i in tuning)}
    subsets = []
    for size in range(1, 5):
        for indices in combinations(range(4), size):
            selected = min(indices, key=lambda j: (-metrics[names[j]]['average_precision'], -metrics[names[j]]['budget_recall'], names[j]))
            subsets.append({'healthy_models': [names[j] for j in indices],
                            'selected_model': names[selected],
                            'weights': [float(j == selected) for j in range(4)],
                            'threshold': float(sorted(matrix[freeze,selected], reverse=True)[math.floor(budget*len(freeze))])})
    payload = {'schema_version': '0.6-resilience-policy', 'model_names': names,
               'budget_fraction': budget, 'tuning_row_ids': [ids[i] for i in tuning],
               'policy_row_ids': [ids[i] for i in freeze],
               'model_selection_metrics': metrics, 'subsets': subsets}
    # This binds a local policy to its fitted payload against accidental edits.
    # It is not an independent signature: someone can rewrite and rehash it.
    return {**payload, 'payload_sha256': _sha(payload)}


class DeliveryError(ValueError):
    """Malformed or corrupted score delivery; a fresh inference may recover."""


class ModelUnavailableError(RuntimeError):
    """Predictor explicitly reports temporary unavailability."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_policy(policy):
    """Reject an inconsistent fallback before any predictor is invoked."""
    expected_keys = {'schema_version', 'model_names', 'budget_fraction',
                     'tuning_row_ids', 'policy_row_ids', 'model_selection_metrics',
                     'subsets', 'payload_sha256'}
    if type(policy) is not dict or set(policy) != expected_keys or policy['schema_version'] != '0.6-resilience-policy':
        raise ValueError('invalid_resilience_policy')
    names = policy['model_names']
    subsets = policy['subsets']
    if (type(names) is not list or len(names) != 4 or
            any(type(name) is not str or not name.strip() for name in names) or
            len(set(names)) != 4 or type(subsets) is not list or len(subsets) != 15):
        raise ValueError('invalid_resilience_policy')
    _fraction(policy['budget_fraction'])
    tuning = policy['tuning_row_ids']
    freeze = policy['policy_row_ids']
    if type(tuning) is not list or type(freeze) is not list or not tuning or not freeze:
        raise ValueError('invalid_resilience_policy')
    tuning_ids = _row_ids(tuning, len(tuning))
    freeze_ids = _row_ids(freeze, len(freeze))
    if set(tuning_ids) & set(freeze_ids):
        raise ValueError('resilience_policy_period_overlap')
    metrics = policy['model_selection_metrics']
    if type(metrics) is not dict or set(metrics) != set(names):
        raise ValueError('invalid_resilience_policy')
    for name in names:
        item = metrics[name]
        if type(item) is not dict or set(item) != {'average_precision', 'budget_recall'}:
            raise ValueError('invalid_resilience_policy')
        _probabilities([item['average_precision'], item['budget_recall']], 'policy_metrics')
    expected_subsets = [[names[j] for j in indices]
                        for size in range(1, 5)
                        for indices in combinations(range(4), size)]
    for subset, expected_healthy in zip(subsets, expected_subsets):
        if type(subset) is not dict or set(subset) != {'healthy_models', 'selected_model', 'weights', 'threshold'}:
            raise ValueError('invalid_resilience_policy')
        healthy = subset['healthy_models']
        selected = subset['selected_model']
        weights = subset['weights']
        if type(weights) is not list or len(weights) != 4:
            raise ValueError('invalid_resilience_policy')
        _probabilities(weights, 'policy_weights')
        _probabilities([subset['threshold']], 'policy_threshold')
        if (type(healthy) is not list or healthy != expected_healthy or
                type(selected) is not str or selected not in healthy or
                weights != [float(name == selected) for name in names]):
            raise ValueError('invalid_resilience_policy')
        winner = min(healthy, key=lambda name: (-metrics[name]['average_precision'],
                                                -metrics[name]['budget_recall'], name))
        if selected != winner:
            raise ValueError('resilience_policy_selection_mismatch')
    digest = policy.get('payload_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('invalid_resilience_policy')
    payload = {key: value for key, value in policy.items() if key != 'payload_sha256'}
    try:
        if _sha(payload) != digest:
            raise ValueError('resilience_policy_digest_mismatch')
    except (TypeError, OverflowError) as error:
        raise ValueError('invalid_resilience_policy') from error
    return names


def make_receipt(delivery):
    """Return immutable canonical bytes with identity-sorted ID/score pairs."""
    try:
        scores = _probabilities(delivery['scores'])
        ids = _row_ids(delivery['row_ids'], len(scores))
        return _canonical({'context': delivery['context'], 'model': delivery['model'],
                           'records': sorted(zip(ids, scores))})
    except (TypeError, ValueError, KeyError) as error:
        raise DeliveryError('invalid_producer_payload') from error


def validate_delivery(receipt, delivery, row_ids, context, model):
    """Validate a producer receipt and return scores in requested row order."""
    try:
        scores = _probabilities(delivery['scores'])
        ids = _row_ids(delivery['row_ids'], len(scores))
        expected = _row_ids(row_ids, len(scores))
        if set(ids) != set(expected):
            raise DeliveryError('row_identity_mismatch')
        if _canonical(delivery['context']) != _canonical(context) or delivery['model'] != model:
            raise DeliveryError('context_or_model_mismatch')
        if make_receipt(delivery) != receipt:
            raise DeliveryError('receipt_mismatch')
        joined = dict(zip(ids, scores))
        return [joined[i] for i in expected]
    except DeliveryError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise DeliveryError('invalid_delivery') from error


SCENARIOS = ('clean', 'strongest_model_dropout', 'one_model_row_roll',
             'eight_bit_quantization', 'persistent_strongest_model_dropout',
             'reordered_records', 'stale_context', 'all_models_unavailable')


def _inject(delivery, scenario, model, strongest, attempt):
    if scenario == 'reordered_records':
        delivery['row_ids'].reverse()
        delivery['scores'].reverse()
    elif scenario == 'stale_context' and model == strongest and attempt == 1:
        delivery['context'] = {'stale_context': delivery['context']}
    elif model == strongest and (attempt == 1 or scenario == 'persistent_strongest_model_dropout'):
        if scenario in ('strongest_model_dropout', 'persistent_strongest_model_dropout'):
            delivery['scores'] = [0.] * len(delivery['scores'])
        elif scenario == 'one_model_row_roll':
            delivery['scores'] = np.roll(delivery['scores'], 1).tolist()
    if scenario == 'eight_bit_quantization' and attempt == 1:
        delivery['scores'] = (np.round(np.asarray(delivery['scores']) * 255) / 255).tolist()


def _study(produce, row_ids, context, policy, baseline_weights, baseline_threshold):
    names = policy['model_names']
    ids = _row_ids(row_ids, len(row_ids))
    if not ids:
        raise ValueError('row_ids must be nonempty')
    context = json.loads(_canonical(context))
    weights = _probabilities(baseline_weights, 'baseline_weights')
    if len(weights) != 4 or not math.isclose(sum(weights), 1., abs_tol=1e-12):
        raise ValueError('baseline_weights must be four probabilities summing to one')
    apply_policy([.5] * len(ids), ids, baseline_threshold, policy['budget_fraction'])
    strongest = names[int(np.argmax(weights))]
    scenarios = []
    for scenario in SCENARIOS:
        attempts, healthy, quarantined = [], {}, []
        counts = dict.fromkeys(names, 0)
        for model in names:
            def deliver():
                counts[model] += 1
                event = {'model': model, 'attempt': counts[model], 'status': 'accepted',
                         'failure_reason': None, 'receipt_sha256': None, 'delivered_sha256': None}
                delivery = None
                try:
                    produced = produce(model)
                    if scenario == 'all_models_unavailable':
                        raise ModelUnavailableError('injected_unavailable_after_inference')
                    delivery = {'context': context, 'model': model, 'row_ids': list(ids),
                                'scores': np.asarray(produced).tolist()}
                    receipt = make_receipt(delivery)
                    event['receipt_sha256'] = hashlib.sha256(receipt).hexdigest()
                    _inject(delivery, scenario, model, strongest, counts[model])
                    event['delivered_sha256'] = _sha(delivery)
                    scores = validate_delivery(receipt, delivery, ids, context, model)
                except (DeliveryError, ModelUnavailableError) as error:
                    event.update(status='rejected', failure_reason=str(error))
                    if delivery is not None:
                        # Nonfinite malformed producer values are encoded as strings,
                        # keeping failure evidence strict JSON rather than NaN tokens.
                        event['delivery'] = json.loads(json.dumps(delivery, default=str), parse_constant=str)
                    attempts.append(event)
                    raise
                attempts.append(event)
                return scores
            try:
                healthy[model] = Retrying(retry=retry_if_exception_type((DeliveryError, ModelUnavailableError)),
                                          stop=stop_after_attempt(2), wait=wait_none(), reraise=True)(deliver)
            except (DeliveryError, ModelUnavailableError):
                quarantined.append(model)
        healthy_names = [n for n in names if n in healthy]
        result = {'name': scenario, 'status': 'ok', 'route': 'baseline_fusion',
                  'healthy_models': healthy_names, 'quarantined_models': quarantined,
                  'selected_model': None, 'weights': weights, 'threshold': float(baseline_threshold),
                  'scores': [], 'queue': None, 'decision_count': 0, 'coverage': 0.,
                  'attempts': attempts, 'call_counts': counts}
        if len(healthy) < 2:
            result.update(status='fail_closed', route='fail_closed', weights=[], threshold=None)
        else:
            if len(healthy) == 4:
                scores = (np.asarray([healthy[n] for n in names]).T @ np.asarray(weights)).tolist()
            else:
                subset = next(s for s in policy['subsets'] if s['healthy_models'] == healthy_names)
                result.update(route='validation_frozen_fallback', selected_model=subset['selected_model'],
                              weights=subset['weights'], threshold=subset['threshold'])
                scores = healthy[subset['selected_model']]
            result.update(scores=scores, coverage=1., decision_count=len(ids),
                          queue=apply_policy(scores, ids, result['threshold'], policy['budget_fraction']))
        scenarios.append(result)
    return {'schema_version': '0.6-resilience', 'context': context, 'row_ids': ids,
            'strongest_model': strongest, 'scenarios': scenarios}


def run_resilience_study(predictors, features, row_ids, context, policy,
                         baseline_weights, baseline_threshold):
    """Execute fresh predictions for every attempted delivery, without labels."""
    names = _validate_policy(policy)
    if not isinstance(predictors, Mapping) or set(predictors) != set(names) or any(not callable(predictors[n]) for n in names):
        raise ValueError('predictors must contain exactly the four model callbacks')
    values = np.asarray(features)
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1] or
            values.dtype.kind not in 'fiu' or
            not np.isfinite(values).all()):
        raise ValueError('features must be a nonempty numeric matrix aligned with row_ids')
    ids = _row_ids(row_ids, values.shape[0])
    return _study(lambda model: predictors[model](features), ids, context, policy,
                  baseline_weights, baseline_threshold)


def reconstruct_resilience(reference_scores, row_ids, context, policy,
                           baseline_weights, baseline_threshold):
    """Rebuild expected deterministic evidence from archived scores; no models.

    This is a verifier calculation, never proof that a predictor ran. Runtime
    execution is supplied exclusively through run_resilience_study callbacks.
    """
    names = _validate_policy(policy)
    if isinstance(reference_scores, (str, bytes, Mapping, Set)):
        raise ValueError('reference_scores must be a nonempty aligned n-by-4 matrix')
    try:
        rows = [_probabilities(row) for row in reference_scores]
    except TypeError as error:
        raise ValueError('reference_scores must be a nonempty aligned n-by-4 matrix') from error
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError('reference_scores must be a nonempty aligned n-by-4 matrix')
    ids = _row_ids(row_ids, len(rows))
    columns = {name: [row[j] for row in rows] for j, name in enumerate(names)}
    return _study(lambda model: columns[model], ids, context, policy,
                  baseline_weights, baseline_threshold)
