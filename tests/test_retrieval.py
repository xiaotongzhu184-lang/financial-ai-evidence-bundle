"""Contract tests for source-declared, extractive financial retrieval."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from yk_evidence.retrieval import run_retrieval, verify_retrieval


def fixture_corpus() -> dict:
    return {
        "schema_version": "0.4",
        "documents": [
            {
                "id": "acme_fy2025_revenue",
                "title": "Acme fiscal 2025 revenue fact (test fixture)",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                "classification": "test_fixture_not_a_real_sec_disclosure",
                "text": "Acme revenue was USD 100 for fiscal year 2025.",
            },
            {
                "id": "beta_fy2025_cash",
                "title": "Beta fiscal 2025 cash flow fact (test fixture)",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json",
                "classification": "test_fixture_not_a_real_sec_disclosure",
                "text": "Beta net cash provided by operating activities was USD 50 in fiscal 2025.",
            },
        ],
    }


class RetrievalContractTest(unittest.TestCase):
    def test_retrieves_literal_cited_excerpt_and_verifies_saved_rankings(self):
        corpus = fixture_corpus()
        evidence = run_retrieval(corpus, ["Acme revenue fiscal 2025"])

        self.assertEqual("tfidf_extractive", evidence["method"])
        self.assertEqual("supported", evidence["queries"][0]["status"])
        answer = evidence["queries"][0]
        self.assertEqual("Acme revenue was USD 100 for fiscal year 2025.", answer["answer_excerpt"])
        self.assertEqual("acme_fy2025_revenue", answer["citation"]["document_id"])
        self.assertEqual("acme_fy2025_revenue:chunk:0000", answer["citation"]["chunk_id"])
        self.assertEqual(corpus["documents"][0]["source_url"], answer["citation"]["source_url"])
        self.assertEqual(1, answer["ranked_results"][0]["rank"])
        self.assertGreater(answer["ranked_results"][0]["score"], 0)
        self.assertEqual([], verify_retrieval(corpus, evidence))

    def test_no_vocabulary_overlap_abstains_without_citation(self):
        evidence = run_retrieval(fixture_corpus(), ["unicorn zeppelin nebula"])
        answer = evidence["queries"][0]
        self.assertEqual("insufficient_evidence", answer["status"])
        self.assertEqual("no_indexed_term_overlap", answer["reason"])
        self.assertIsNone(answer["answer_excerpt"])
        self.assertIsNone(answer["citation"])
        self.assertEqual([], answer["ranked_results"])

    def test_second_chunk_keeps_literal_offsets_and_a_stable_id(self):
        corpus = fixture_corpus()
        corpus["documents"][0]["text"] = ("alpha " * 80) + "quasar revenue USD 77"
        answer = run_retrieval(corpus, ["quasar revenue"])["queries"][0]
        citation = answer["citation"]
        self.assertEqual("acme_fy2025_revenue:chunk:0001", citation["chunk_id"])
        self.assertEqual("quasar revenue USD 77", citation["excerpt"])
        self.assertEqual(
            citation["excerpt"],
            corpus["documents"][0]["text"][citation["start_char"] : citation["end_char"]],
        )

    def test_known_issuer_does_not_make_unknown_metric_or_year_answerable(self):
        evidence = run_retrieval(
            fixture_corpus(),
            ["Acme employee count fiscal 2025", "Acme revenue fiscal 2035"],
        )
        for answer in evidence["queries"]:
            self.assertEqual("insufficient_evidence", answer["status"])
            self.assertEqual("query_terms_not_in_cited_excerpt", answer["reason"])
            self.assertIsNone(answer["answer_excerpt"])
            self.assertIsNone(answer["citation"])
            self.assertTrue(answer["ranked_results"])

    def test_duplicate_document_ids_are_rejected(self):
        corpus = fixture_corpus()
        corpus["documents"][1]["id"] = corpus["documents"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate document id"):
            run_retrieval(corpus, ["revenue"])

    def test_unsafe_source_urls_and_ids_are_rejected(self):
        for unsafe_url in (
            "file:///tmp/secret.txt",
            "../secret.txt",
            "http://localhost/private",
            "https://data.sec.gov/%2e%2e/secret.txt",
        ):
            with self.subTest(unsafe_url=unsafe_url):
                corpus = fixture_corpus()
                corpus["documents"][0]["source_url"] = unsafe_url
                with self.assertRaisesRegex(ValueError, "source_url"):
                    run_retrieval(corpus, ["revenue"])
        corpus = fixture_corpus()
        corpus["documents"][0]["id"] = "../secret"
        with self.assertRaisesRegex(ValueError, "document id"):
            run_retrieval(corpus, ["revenue"])

    def test_empty_corpus_text_or_question_is_rejected(self):
        corpus = fixture_corpus()
        corpus["documents"][0]["text"] = "   "
        with self.assertRaisesRegex(ValueError, "text"):
            run_retrieval(corpus, ["revenue"])
        with self.assertRaisesRegex(ValueError, "question"):
            run_retrieval(fixture_corpus(), ["  "])

    def test_rehashed_citation_score_or_answer_tampering_fails_semantic_verification(self):
        corpus = fixture_corpus()
        original = run_retrieval(corpus, ["Acme revenue fiscal 2025"])
        for change in ("score", "source_url", "answer_excerpt", "chunk_sha256"):
            with self.subTest(change=change):
                evidence = copy.deepcopy(original)
                item = evidence["queries"][0]
                if change == "score":
                    item["ranked_results"][0]["score"] += 0.1
                elif change == "source_url":
                    item["citation"]["source_url"] = "https://example.org/false"
                elif change == "answer_excerpt":
                    item["answer_excerpt"] = "Acme revenue was USD 999."
                else:
                    item["citation"]["chunk_sha256"] = "0" * 64
                self.assertTrue(verify_retrieval(corpus, evidence))

    def test_changed_archived_corpus_fails_verification(self):
        corpus = fixture_corpus()
        evidence = run_retrieval(corpus, ["Acme revenue fiscal 2025"])
        changed = copy.deepcopy(corpus)
        changed["documents"][0]["text"] = "Acme revenue was USD 999 for fiscal year 2025."
        self.assertTrue(verify_retrieval(changed, evidence))

    def test_packaged_sec_corpus_retrieves_three_distinct_fy2025_revenue_facts(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src/yk_evidence/assets/data/financial/corpus_v04_sec_companyfacts.json"
        )
        corpus = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(3, len(corpus["documents"]))
        for document in corpus["documents"]:
            self.assertEqual("public_sec_companyfacts_derived", document["classification"])
            self.assertEqual(64, len(document["source_record"]["raw_response_sha256"]))
            self.assertEqual(2, len(document["source_record"]["facts"]))
            self.assertTrue(all(str(fact["val"]) in document["text"] for fact in document["source_record"]["facts"]))
        evidence = run_retrieval(
            corpus,
            ["Microsoft fiscal 2025 revenue", "Apple fiscal 2025 revenue", "NVIDIA fiscal 2025 revenue"],
        )
        self.assertEqual(
            ["msft_fy2025_revenue", "aapl_fy2025_revenue", "nvda_fy2025_revenue"],
            [item["citation"]["document_id"] for item in evidence["queries"]],
        )
        self.assertEqual([], verify_retrieval(corpus, evidence))


if __name__ == "__main__":
    unittest.main()
