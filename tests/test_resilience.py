import copy
import hashlib
import json
import numpy as np
import unittest

from yk_evidence import resilience as r

NAMES = ['a', 'b', 'c', 'd']
IDS = list(range(8))
MATRIX = np.array([[.1,.8,.2,.4],[.9,.2,.7,.5]] * 4)
LABELS = [0,1] * 4


def policy():
    return r.fit_resilience_policy(MATRIX, LABELS, IDS, IDS, NAMES, .25)


def rehashed_policy(change):
    frozen = copy.deepcopy(policy())
    change(frozen)
    payload = {key: value for key, value in frozen.items() if key != 'payload_sha256'}
    frozen['payload_sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')).hexdigest()
    return frozen


class ResilienceTests(unittest.TestCase):
    def test_runtime_rejects_quarantined_weight_policy_before_prediction(self):
        frozen = policy()
        fallback = next(s for s in frozen['subsets'] if s['healthy_models'] == ['b', 'c', 'd'])
        fallback['weights'] = [1., 0., 0., 0.]
        fallback['threshold'] = .01
        calls = []
        predictors = {name: (lambda _, name=name, column=j:
                             (calls.append(name), MATRIX[:, column])[1])
                      for j, name in enumerate(NAMES)}
        with self.assertRaises(ValueError):
            r.run_resilience_study(predictors, MATRIX, IDS, {'batch': 'x'}, frozen,
                                   [1., 0., 0., 0.], .5)
        assert calls == []

    def test_unchanged_policy_receipt_is_required_before_prediction_or_reconstruction(self):
        frozen = policy()
        fallback = next(s for s in frozen['subsets'] if s['healthy_models'] == ['b', 'c', 'd'])
        fallback['threshold'] = .01  # still a valid probability, but no longer the fitted value
        calls = []
        predictors = {name: (lambda _, name=name, column=j:
                             (calls.append(name), MATRIX[:, column])[1])
                      for j, name in enumerate(NAMES)}
        with self.assertRaises(ValueError):
            r.run_resilience_study(predictors, MATRIX, IDS, {'batch': 'x'}, frozen,
                                   [1., 0., 0., 0.], .5)
        assert calls == []
        with self.assertRaises(ValueError):
            r.reconstruct_resilience(MATRIX, IDS, {'batch': 'x'}, frozen,
                                     [1., 0., 0., 0.], .5)

    def test_rehashed_policy_requires_exact_schema_and_canonical_subsets(self):
        cases = {
            'extra_top_level': lambda p: p.update({'ignored': 1}),
            'missing_schema': lambda p: p.pop('schema_version'),
            'bad_schema': lambda p: p.update(schema_version='other'),
            'duplicate_names': lambda p: p['model_names'].__setitem__(1, 'a'),
            'unhashable_name': lambda p: p['model_names'].__setitem__(1, ['b']),
            'missing_subset': lambda p: p['subsets'].pop(),
            'wrong_subset_order': lambda p: p['subsets'].reverse(),
            'wrong_healthy_order': lambda p: p['subsets'][4]['healthy_models'].reverse(),
            'extra_subset_field': lambda p: p['subsets'][0].update({'ignored': 1}),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                frozen = rehashed_policy(change)
                calls = []
                predictors = {name: (lambda _, name=name, column=j:
                                     (calls.append(name), MATRIX[:, column])[1])
                              for j, name in enumerate(NAMES)}
                with self.assertRaises(ValueError):
                    r.run_resilience_study(predictors, MATRIX, IDS, {}, frozen,
                                           [1., 0., 0., 0.], .5)
                assert calls == []

    def test_rehashed_policy_rejects_invalid_ids_metrics_threshold_and_weights(self):
        cases = {
            'duplicate_tuning_id': lambda p: p['tuning_row_ids'].__setitem__(1, 0),
            'overlapping_periods': lambda p: p['policy_row_ids'].__setitem__(0, 0),
            'boolean_period_id': lambda p: p['policy_row_ids'].__setitem__(0, True),
            'invalid_budget': lambda p: p.update(budget_fraction=1.),
            'boolean_budget': lambda p: p.update(budget_fraction=True),
            'missing_metric': lambda p: p['model_selection_metrics'].pop('a'),
            'extra_metric_field': lambda p: p['model_selection_metrics']['a'].update({'ignored': 0}),
            'invalid_ap': lambda p: p['model_selection_metrics']['a'].update(average_precision=1.1),
            'boolean_recall': lambda p: p['model_selection_metrics']['a'].update(budget_recall=False),
            'invalid_threshold': lambda p: p['subsets'][0].update(threshold=1.1),
            'boolean_threshold': lambda p: p['subsets'][0].update(threshold=True),
            'boolean_weight': lambda p: p['subsets'][0].update(weights=[True, False, False, False]),
            'quarantined_weight': lambda p: p['subsets'][13].update(weights=[1., 0., 0., 0.]),
            'not_metric_winner': lambda p: p['subsets'][10].update(selected_model='b',
                                                                   weights=[0., 1., 0., 0.]),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                frozen = rehashed_policy(change)
                calls = []
                predictors = {name: (lambda _, name=name, column=j:
                                     (calls.append(name), MATRIX[:, column])[1])
                              for j, name in enumerate(NAMES)}
                with self.assertRaises(ValueError):
                    r.run_resilience_study(predictors, MATRIX, IDS, {}, frozen,
                                           [1., 0., 0., 0.], .5)
                assert calls == []

    def test_runtime_rejects_bad_public_inputs_without_calling_predictors(self):
        calls = []
        predictors = {name: (lambda _, name=name, column=j:
                             (calls.append(name), MATRIX[:, column])[1])
                      for j, name in enumerate(NAMES)}
        cases = [
            (predictors, np.full((8, 4), float('nan'))),
            (None, MATRIX),
            (predictors, np.ones((8, 4), dtype=bool)),
        ]
        for callbacks, features in cases:
            with self.subTest(callbacks=callbacks is not None, dtype=np.asarray(features).dtype):
                calls.clear()
                with self.assertRaises(ValueError):
                    r.run_resilience_study(callbacks, features, IDS, {}, policy(),
                                           [1., 0., 0., 0.], .5)
                assert calls == []

    def test_public_run_and_reconstruction_reject_missing_row_ids_as_value_error(self):
        calls = []
        predictors = {name: (lambda _, name=name, column=j:
                             (calls.append(name), MATRIX[:, column])[1])
                      for j, name in enumerate(NAMES)}
        with self.assertRaises(ValueError):
            r.run_resilience_study(predictors, MATRIX, None, {}, policy(),
                                   [1., 0., 0., 0.], .5)
        assert calls == []
        with self.assertRaises(ValueError):
            r.reconstruct_resilience(MATRIX, None, {}, policy(),
                                     [1., 0., 0., 0.], .5)

    def test_fit_requires_four_explicit_distinct_model_names(self):
        for names in (None, 'abcd', ['a', ['b'], 'c', 'd']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                r.fit_resilience_policy(MATRIX, LABELS, IDS, IDS, names, .25)

    def test_reconstruction_rejects_malformed_reference_score_input(self):
        for scores in (None, np.ones((8, 4), dtype=bool), np.ones((8, 3))):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                r.reconstruct_resilience(scores, IDS, {}, policy(),
                                         [1., 0., 0., 0.], .5)

    def test_malformed_callback_scores_are_retried_then_quarantined(self):
        for broken_scores in (np.ones((8, 1)), np.ones(8, dtype=bool), np.ones(7)):
            with self.subTest(shape=broken_scores.shape, dtype=broken_scores.dtype):
                predictors = {name: (lambda _, column=j: MATRIX[:, column])
                              for j, name in enumerate(NAMES)}
                predictors['a'] = lambda _, value=broken_scores: value
                report = r.run_resilience_study(predictors, MATRIX, IDS, {}, policy(),
                                                [1., 0., 0., 0.], .5)
                clean = report['scenarios'][0]
                assert clean['quarantined_models'] == ['a']
                assert clean['call_counts']['a'] == 2
                assert clean['scores'] == [.2, .7] * 4

    def test_validation_selection_and_separate_threshold_period(self):
        p = policy()
        assert len(p['subsets']) == 15
        assert p['tuning_row_ids'] == [0,1,2,3]
        assert p['policy_row_ids'] == [4,5,6,7]
        item = next(s for s in p['subsets'] if s['healthy_models'] == ['b','c','d'])
        assert item['selected_model'] == 'c'
        assert item['weights'] == [0.,0.,1.,0.]
        assert item['threshold'] == .7

    def test_receipt_is_immutable_and_records_align_by_identity(self):
        delivery = {'context': {'batch': 'x'}, 'model': 'a', 'row_ids': [2,1], 'scores': [.9,.1]}
        receipt = r.make_receipt(delivery)
        delivery['row_ids'].reverse()
        delivery['scores'].reverse()
        assert r.validate_delivery(receipt, delivery, [1,2], {'batch':'x'}, 'a') == [.1,.9]
        delivery['scores'][0] = 0.
        with self.assertRaises(r.DeliveryError):
            r.validate_delivery(receipt, delivery, [1,2], {'batch':'x'}, 'a')

    def test_receipt_rejects_bad_identity_context_and_values(self):
        clean = {'context': {'batch': 'x'}, 'model': 'a', 'row_ids': [1,2], 'scores': [.1,.9]}
        receipt = r.make_receipt(clean)
        for changes in ({'row_ids':[1,1]}, {'row_ids':[1,3]}, {'context':{'batch':'stale'}},
                        {'scores':[float('nan'),.9]}, {'scores':[-.1,.9]}, {'scores':[0.,0.]}, {'model':'b'}):
            with self.subTest(changes=changes), self.assertRaises(r.DeliveryError):
                r.validate_delivery(receipt, dict(clean, **changes), [1,2], {'batch':'x'}, 'a')

    def test_study_retries_real_predictors_and_reconstructs_transcript(self):
        counts = dict.fromkeys(NAMES, 0)
        def predictor(j):
            def predict(features):
                assert features is MATRIX
                counts[NAMES[j]] += 1
                return features[:,j]
            return predict
        report = r.run_resilience_study({n:predictor(j) for j,n in enumerate(NAMES)}, MATRIX, IDS,
                                        {'batch':'x'}, policy(), [1.,0.,0.,0.], .5)
        by = {s['name']:s for s in report['scenarios']}
        assert len(by) == 8
        for name in ['clean','strongest_model_dropout','one_model_row_roll','eight_bit_quantization','reordered_records','stale_context']:
            assert by[name]['scores'] == [.1,.9]*4
            assert by[name]['coverage'] == 1.
        assert by['strongest_model_dropout']['call_counts']['a'] == 2
        assert by['one_model_row_roll']['attempts'][0]['status'] == 'rejected'
        assert by['reordered_records']['call_counts']['a'] == 1
        persistent = by['persistent_strongest_model_dropout']
        assert persistent['quarantined_models'] == ['a']
        assert persistent['selected_model'] == 'c'
        assert persistent['scores'] == [.2,.7]*4
        assert persistent['threshold'] == .7
        failed = by['all_models_unavailable']
        assert failed['status'] == 'fail_closed'
        assert failed['scores'] == [] and failed['queue'] is None
        assert failed['decision_count'] == 0 and failed['coverage'] == 0
        assert all(v == 2 for v in failed['call_counts'].values())
        assert counts == {n:sum(s['call_counts'][n] for s in report['scenarios']) for n in NAMES}
        assert report == r.reconstruct_resilience(MATRIX, IDS, {'batch':'x'}, policy(), [1.,0.,0.,0.], .5)

    def test_retry_uses_new_callback_values_and_does_not_swallow_programming_errors(self):
        counts = dict.fromkeys(NAMES, 0)
        def callback(j):
            def predict(_):
                counts[NAMES[j]] += 1
                return MATRIX[:,j] if counts[NAMES[j]] < 3 else np.array([.3,.8]*4)
            return predict
        output = r.run_resilience_study({n:callback(j) for j,n in enumerate(NAMES)}, MATRIX, IDS,
                                       {'batch':'x'}, policy(), [1.,0.,0.,0.], .5)
        assert output['scenarios'][1]['scores'] == [.3,.8]*4
        def broken(_):
            raise TypeError('programming defect')
        with self.assertRaisesRegex(TypeError, 'programming defect'):
            r.run_resilience_study(dict.fromkeys(NAMES, broken), MATRIX, IDS, {}, policy(), [1.,0.,0.,0.], .5)

    def test_fewer_than_two_healthy_withholds_every_row_and_quarantine_is_per_scenario(self):
        def unavailable(_):
            raise r.ModelUnavailableError('external_unavailable')
        predictors = dict.fromkeys(NAMES, unavailable)
        predictors['a'] = lambda _: MATRIX[:,0]
        output = r.run_resilience_study(predictors, MATRIX, IDS, {}, policy(), [1.,0.,0.,0.], .5)
        for scenario in output['scenarios']:
            assert scenario['status'] == 'fail_closed'
            assert scenario['scores'] == [] and scenario['decision_count'] == 0
            assert scenario['call_counts']['b'] == 2
        clean = r.run_resilience_study({n:(lambda _,j=j:MATRIX[:,j]) for j,n in enumerate(NAMES)}, MATRIX, IDS,
                                      {}, policy(), [1.,0.,0.,0.], .5)
        assert clean['scenarios'][0]['status'] == 'ok'

    def test_runtime_rejects_bad_input_and_quarantines_invalid_prediction(self):
        predictors = {n:(lambda _,j=j:MATRIX[:,j]) for j,n in enumerate(NAMES)}
        for features, ids, callbacks in [(MATRIX, [1]*8, predictors), (MATRIX[:2], IDS, predictors),
                                         (np.ones(8), IDS, predictors), (MATRIX, IDS, {'a': predictors['a']}),
                                         (np.full((8,4), np.inf), IDS, predictors)]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                r.run_resilience_study(callbacks, features, ids, {}, policy(), [1.,0.,0.,0.], .5)
        predictors['a'] = lambda _: [float('nan')]*8
        output = r.run_resilience_study(predictors, MATRIX, IDS, {}, policy(), [1.,0.,0.,0.], .5)
        assert output['scenarios'][0]['quarantined_models'] == ['a']
        assert output['scenarios'][0]['scores'] == [.2,.7]*4

    def test_policy_selection_does_not_consume_second_half_labels(self):
        flipped = LABELS[:4] + [1-v for v in LABELS[4:]]
        assert policy() == r.fit_resilience_policy(MATRIX, flipped, IDS, IDS, NAMES, .25)
