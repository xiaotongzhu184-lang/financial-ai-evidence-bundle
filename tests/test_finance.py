import copy
import unittest
from pathlib import Path

from yk_evidence.finance import answer_revenue_question, load_snapshot


SNAPSHOT = Path(__file__).resolve().parents[1] / "data/financial/msft_2025_10k_revenue.json"


class FinanceSnapshotTest(unittest.TestCase):
    def test_revenue_change_has_two_exact_sec_citations(self):
        source = load_snapshot(SNAPSHOT)
        memo = answer_revenue_question(source)
        self.assertEqual("supported", memo["status"])
        self.assertEqual(36602000000, memo["increase_usd"])
        self.assertEqual("0000950170-25-100235", memo["accession"])
        self.assertEqual({"2024-06-30", "2025-06-30"}, {x["period_end"] for x in memo["citations"]})
        self.assertTrue(all(x["source_sha256"] == source["raw_source_sha256"] for x in memo["citations"]))

    def test_mismatched_accession_is_rejected(self):
        source = load_snapshot(SNAPSHOT)
        changed = copy.deepcopy(source)
        changed["facts"][0]["accn"] = "different-accession"
        with self.assertRaisesRegex(ValueError, "accession"):
            answer_revenue_question(changed)

    def test_mismatched_units_are_rejected(self):
        source = load_snapshot(SNAPSHOT)
        changed = copy.deepcopy(source)
        changed["facts"][1]["unit"] = "USD/shares"
        with self.assertRaisesRegex(ValueError, "unit"):
            answer_revenue_question(changed)


if __name__ == "__main__":
    unittest.main()
