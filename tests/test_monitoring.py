"""Behavioral checks for the attributed PSI core and frozen score windows."""

from __future__ import annotations

import unittest

import numpy as np

from yk_evidence._vendor.evidently_psi import psi_components
from yk_evidence.monitoring import monitor_score_windows


class MonitoringTests(unittest.TestCase):
    def test_constant_internal_reference_is_separate_from_both_domain_endpoints(self) -> None:
        """Catches a constant reference sharing one bin with an extreme later score."""
        for reference_value, expected_edges in (
            (.25, [0.0, .125, .625, 1.0]),
            (.75, [0.0, .375, .875, 1.0]),
        ):
            for later_value, expected_bin in ((0.0, 0), (1.0, 2)):
                with self.subTest(reference=reference_value, later=later_value):
                    result = monitor_score_windows(
                        [reference_value] * 3, [1, 2, 3], [later_value] * 6,
                        [0, 0, 1, 1, 2, 2], [4, 5, 6, 7, 8, 9], [1, 0] * 3,
                        .5, .5,
                    )
                    self.assertEqual(expected_edges, result["reference"]["bin_edges"])
                    self.assertEqual([0, 3, 0], result["reference"]["bin_counts"])
                    for window in result["windows"]:
                        self.assertEqual(2, window["bin_counts"][expected_bin])
                        self.assertGreater(window["psi"], 0)

    def test_psi_core_is_symmetric_finite_and_rejects_unprotected_zeroes(self) -> None:
        """Catches unstable zero handling or asymmetric alteration of the upstream core."""
        left = psi_components([.2, .3, .5], [.1, .4, .5])
        right = psi_components([.1, .4, .5], [.2, .3, .5])
        self.assertAlmostEqual(left["psi"], right["psi"], places=14)
        self.assertGreaterEqual(left["psi"], 0)
        for invalid in ([0, .5, .5], [float("nan"), .3, .7], [float("inf"), .3, .7]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                psi_components(invalid, [.2, .3, .5])

    def test_frozen_reference_bins_do_not_shift_with_test_distribution(self) -> None:
        """Catches deriving histogram edges from the later period being monitored."""
        common = ([.1, .5, .9], [100, 101, 102])
        times = [0, 0, 1, 1, 2, 2]
        ids = [200, 201, 202, 203, 204, 205]
        labels = [1, 0] * 3
        identity = monitor_score_windows(*common, [.1, .5] * 3, times, ids, labels, .4, .5)
        shifted = monitor_score_windows(*common, [1] * 6, times, ids, labels, .4, .5)
        self.assertEqual(identity["reference"]["bin_edges"], shifted["reference"]["bin_edges"])
        self.assertEqual(identity["reference"]["bin_counts"], shifted["reference"]["bin_counts"])
        self.assertGreater(shifted["windows"][0]["psi"], identity["windows"][0]["psi"])
        for window in shifted["windows"]:
            self.assertEqual(2, sum(window["bin_counts"]))
            self.assertAlmostEqual(1, sum(window["smoothed_proportions"]))

    def test_monitor_rejects_invalid_grain_ranges_and_unformable_windows(self) -> None:
        """Catches accepting malformed scores, duplicated identities or an empty time block."""
        base = [[.1, .5, .9], [100, 101, 102], [.1, .9] * 3,
                [0, 0, 1, 1, 2, 2], [200, 201, 202, 203, 204, 205], [1, 0] * 3, .4, .5]
        changes = {
            "reference_nan": (0, [.1, float("nan"), .9]),
            "test_out_of_range": (2, [.1, 1.1] * 3),
            "negative_time": (3, [0, 0, -1, 1, 2, 2]),
            "nonfinite_time": (3, [0, 0, float("inf"), 1, 2, 2]),
            "duplicate_test_id": (4, [200, 201, 202, 203, 204, 204]),
            "overlapping_partition_id": (4, [100, 201, 202, 203, 204, 205]),
            "boolean_label": (5, [True, 0] * 3),
            "one_time_group": (3, [0] * 6),
            "zero_budget": (7, 0),
            "boolean_threshold": (6, True),
        }
        for name, (position, replacement) in changes.items():
            case = list(base)
            case[position] = replacement
            with self.subTest(name=name), self.assertRaises(ValueError):
                monitor_score_windows(*case)

    def test_constant_endpoint_reference_still_detects_extreme_score_shift(self) -> None:
        """Catches one-bin collapse hiding a zero-to-one score pipeline failure."""
        result = monitor_score_windows(
            [0, 0, 0], [1, 2, 3], [1] * 6, [0, 0, 1, 1, 2, 2],
            [4, 5, 6, 7, 8, 9], [0] * 6, .5, .5,
        )
        self.assertEqual([0.0, .5, 1.0], result["reference"]["bin_edges"])
        self.assertEqual(2, result["parameters"]["effective_bins"])
        for window in result["windows"]:
            self.assertEqual(2, sum(window["bin_counts"]))
            self.assertGreater(window["psi"], 0)
            self.assertEqual("single_class_unavailable", window["queue_metrics_status"])
            self.assertIsNone(window["queue_metrics"])

    def test_identical_windows_keep_reference_bins_and_time_ties_intact(self) -> None:
        """Catches test-derived bins, dropped endpoints, split timestamps or missing queue limits."""
        result = monitor_score_windows(
            [.1, .5, .9], [100, 101, 102],
            [.1, .5, .9] * 3, [2, 2, 2, 0, 0, 0, 1, 1, 1],
            [206, 207, 208, 200, 201, 202, 203, 204, 205], [1, 0, 0] * 3,
            .4, .5,
        )
        self.assertEqual("score-monitor-1", result["schema_version"])
        reference = result["reference"]
        self.assertEqual(0.0, reference["bin_edges"][0])
        self.assertEqual(1.0, reference["bin_edges"][-1])
        self.assertEqual(10, result["parameters"]["effective_bins"])
        self.assertEqual(3, sum(reference["bin_counts"]))
        self.assertEqual([list(range(200, 203)), list(range(203, 206)), list(range(206, 209))],
                         [window["row_ids"] for window in result["windows"]])
        for window in result["windows"]:
            self.assertEqual(3, window["row_count"])
            self.assertEqual(1, window["positive_count"])
            self.assertEqual(3, sum(window["bin_counts"]))
            self.assertAlmostEqual(0.0, window["psi"], places=14)
            self.assertAlmostEqual(window["psi"], sum(window["psi_contributions"]), places=14)
            self.assertTrue(window["small_sample"])
            self.assertEqual(2, window["queue"]["threshold_alert_count"])
            self.assertEqual(1, window["queue"]["reviewed_count"])
            self.assertEqual(1, window["queue"]["deferred_count"])
            self.assertEqual("measured", window["queue_metrics_status"])

    def test_ported_psi_matches_fixed_upstream_formula_for_positive_proportions(self) -> None:
        """Catches a changed sign, ratio order or aggregation in the copied PSI core."""
        reference = np.array([.2, .3, .5])
        current = np.array([.1, .4, .5])
        result = psi_components(reference, current)
        upstream_terms = (reference - current) * np.log(reference / current)
        self.assertEqual(3, len(result["contributions"]))
        np.testing.assert_allclose(result["contributions"], upstream_terms, rtol=0, atol=1e-14)
        self.assertAlmostEqual(float(np.sum(upstream_terms)), result["psi"], places=14)
        self.assertAlmostEqual(.09808292530117267, result["psi"], places=14)
