"""Worked examples at the public fusion-policy seams."""

from __future__ import annotations

import unittest

from yk_evidence.fusion import apply_policy, assess_scores, fit_fusion


class FusionTests(unittest.TestCase):
    def test_validation_keeps_equal_timestamp_groups_and_ignores_policy_labels_in_selection(self) -> None:
        """Catches splitting a timestamp tie or using policy labels to tune weights."""
        labels = [1, 0] * 8
        times = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4
        scores = [[.8 if label else .2] * 4 for label in labels]
        first = fit_fusion(scores, labels, times, list(range(100, 116)), ["a", "b", "c", "d"], .5)
        flipped_policy = labels[:9] + [1 - label for label in labels[9:]]
        second = fit_fusion(scores, flipped_policy, times, list(range(100, 116)), ["a", "b", "c", "d"], .5)
        self.assertEqual([list(range(100, 106)), list(range(106, 109))], first["tuning_half_row_ids"])
        self.assertEqual(list(range(109, 116)), first["policy_row_ids"])
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["candidates"], second["candidates"])

    def test_public_seams_reject_invalid_inputs_without_numeric_coercion(self) -> None:
        """Catches bool, NaN, duplicate identity and invalid-budget evidence entering a policy."""
        for scores, ids, threshold, budget in (
            ([.8, True], [1, 2], .5, .5),
            ([.8, float("nan")], [1, 2], .5, .5),
            ([.8, .2], [1, 1], .5, .5),
            ([.8, .2], [1, 2], True, .5),
            ([.8, .2], [1, 2], .5, 0),
            ([.8, .2], [1, 2], .5, 1),
            ([.8, .2], [1, 2], .5, True),
        ):
            with self.subTest(scores=scores, ids=ids, threshold=threshold, budget=budget):
                with self.assertRaises(ValueError):
                    apply_policy(scores, ids, threshold, budget)
        with self.assertRaisesRegex(ValueError, "sequence"):
            apply_policy({.9: "first", .2: "second"}, [1, 2], .5, .5)
        with self.assertRaisesRegex(ValueError, "sequence"):
            apply_policy({.9, .2}, [1, 2], .5, .5)
        with self.assertRaises(ValueError):
            assess_scores([1, True], [.8, .2], [1, 2], .5, .5)
        with self.assertRaises(ValueError):
            fit_fusion([[.8, .2, .3, .4]] * 16, [1, 0] * 8,
                       [0, 0, 1, 1] * 4, [1] * 16, ["a", "b", "c", "d"], .5)
        with self.assertRaises(ValueError):
            fit_fusion([[.8, .2, .3, .4]] * 16, [1, 0] * 8,
                       [0, 0, 1, 1] * 4, list(range(16)), ["a", "b", "c", "d"], True)
        with self.assertRaises(ValueError):
            fit_fusion([[.8, .2, .3, .4]] * 16, [1, 0] * 8,
                       [0, 0, 1, 1] * 4, list(range(16)), None, .5)

    def test_weight_search_prioritizes_worst_tuning_period_and_freezes_policy_threshold(self) -> None:
        """Catches optimizing overall ranking while one chronological half misses positives."""
        labels = [1, 0, 1, 0, 0, 1, 0, 1] * 2
        first = [.9, .2, .8, .1, .8, .4, .7, .3]
        second = [.4, .8, .3, .7, .2, .9, .1, .8]
        scores = [[first[i % 8], second[i % 8], .5, .5] for i in range(16)]
        fitted = fit_fusion(scores, labels, [i // 2 for i in range(16)],
                            [100 + i for i in range(16)], ["a", "b", "c", "d"], .5)
        self.assertEqual(35, len(fitted["candidates"]))
        self.assertEqual([.25, .25, 0.0, .5], fitted["weights"])
        self.assertEqual([1.0, 1.0], fitted["candidates"][fitted["selected_candidate_index"]]["half_recalls"])
        self.assertEqual([list(range(100, 104)), list(range(104, 108))], fitted["tuning_half_row_ids"])
        self.assertEqual(list(range(108, 116)), fitted["policy_row_ids"])
        self.assertEqual(.5, fitted["threshold"])
        self.assertEqual(4, fitted["policy_capacity"])
        self.assertEqual(4, fitted["policy_threshold_alert_count"])
        ablations = fitted["ablation_policies"]
        self.assertEqual([.4, .4, .5, .5], [item["threshold"] for item in ablations["single_models"]])
        self.assertEqual(.5, ablations["equal_weight"]["threshold"])
        self.assertEqual(.5, ablations["best_single"]["threshold"])
        self.assertEqual(fitted["threshold"], ablations["robust_weighted"]["threshold"])
        self.assertEqual(0, ablations["best_single"]["reviewed_count"])

    def test_tied_alerts_respect_hard_cap_and_row_id_order(self) -> None:
        """Catches selection beyond capacity or unstable tied-score order."""
        result = apply_policy([.9, .9, .8, .5], [4, 2, 3, 1], .5, .5)
        self.assertEqual([True, True, True, False], result["threshold_alerted"])
        self.assertEqual([True, True, False, False], result["reviewed"])
        self.assertEqual([False, False, True, False], result["deferred"])
        self.assertEqual(2, result["capacity"])
        self.assertEqual(3, result["threshold_alert_count"])
        self.assertEqual(2, result["reviewed_count"])
        self.assertEqual(1, result["deferred_count"])

    def test_assessment_counts_only_reviewed_alerts_as_detections(self) -> None:
        """Catches reporting a deferred positive as a reviewed true positive."""
        result = assess_scores([1, 0, 1, 0], [.9, .8, .4, .2], [1, 2, 3, 4], .3, .5)
        self.assertAlmostEqual(5 / 6, result["average_precision"])
        self.assertAlmostEqual(.2625, result["brier"])
        self.assertEqual({"tp": 1, "fp": 1, "fn": 1, "tn": 1, "rows": 4}, result["counts"])
        self.assertEqual({"tp": 2, "fp": 1, "fn": 0, "tn": 1, "rows": 4}, result["threshold_counts"])
        self.assertEqual(3, result["threshold_alert_count"])
        self.assertEqual(2, result["reviewed_count"])
        self.assertEqual(1, result["deferred_count"])
