# v0.4 extractive financial retrieval protocol

Status: implemented local research slice, 2026-10-11. This protocol describes `yk_evidence.retrieval`, not an LLM or a full Form 10-K retrieval system.

## Source boundary and frozen corpus

The packaged corpus is [`src/yk_evidence/assets/data/financial/corpus_v04_sec_companyfacts.json`](../../src/yk_evidence/assets/data/financial/corpus_v04_sec_companyfacts.json). It contains three short **locally rendered texts derived from structured SEC XBRL Company Facts**, not verbatim passages from the companies' full filings. Each text uses two annual revenue facts from one FY2025 Form 10-K accession: the current fiscal period and its comparison period. It carries the SEC API URL, exact selected fact objects, CIK, taxonomy, concept, unit, accession, filing date and SHA-256 of the full API response received. A local rendering is a cited view of structured fields; its wording is not a quotation from the issuer's prose.

| Issuer | CIK | FY2025 concept | FY2025 period end | FY2025 value, USD | FY2025 10-K accession | Full Company Facts response SHA-256 |
|---|---|---|---|---:|---|---|
| Microsoft | `0000789019` | `RevenueFromContractWithCustomerExcludingAssessedTax` | 2025-06-30 | 281,724,000,000 | `0000950170-25-100235` | `f8aae2965b20ad0df44bdf7ccbedf797d275b6b8dc030154a7a311361bb7246f` |
| Apple | `0000320193` | `RevenueFromContractWithCustomerExcludingAssessedTax` | 2025-09-27 | 416,161,000,000 | `0000320193-25-000079` | `73a86c6aedc31f77cac2ea4df5f80f0b3bd7e6eb58bb4e01444fbedf3afb9c43` |
| NVIDIA | `0001045810` | `Revenues` | 2025-01-26 | 130,497,000,000 | `0001045810-25-000023` | `19ef503a5770f5660964b3c3aea6937579d9b359da344afe6a9adf59c63d26ff` |

Official API endpoints: [Microsoft](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json), [Apple](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json), [NVIDIA](https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json). The [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) specifies that Company Facts aggregates XBRL facts across filings and is updated as filings disseminate. Its Frame API is calendar-oriented and was **not** substituted for issuer fiscal periods here. The [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) sets a 10-requests-per-second maximum and asks for a declared User-Agent. These three responses were fetched serially with a named contact, HTTP 200 and TLS verification success; the exact selected records are in the corpus. `retrieved_utc` records each completed download file's modification time, not an HTTP server timestamp. Complete raw API responses were used to calculate the recorded hashes and are retained outside this repository. The selected fact fields can be checked from the packaged corpus, while a repository-only review cannot recompute the complete-response hashes. A later API fetch may legitimately have a new hash.

The corpus file's raw-byte SHA-256 at creation was `f4634e9167622c814f012479b6f610a05e1d2eeebeab67d7616659aae741c3e9`; the module also records a canonical-JSON corpus digest, `783a7127ea9463d3ab3e925cf676802107d536c8bc771e3ee08106b180aa404c`, in each retrieval result. The two digests have different definitions and are not interchangeable.

## Module interface and deterministic algorithm

```python
from yk_evidence.retrieval import run_retrieval, verify_retrieval
evidence = run_retrieval(corpus: dict, questions: list[str])
errors = verify_retrieval(corpus: dict, evidence: dict)  # [] means local consistency
```

Input corpus version is `0.4`, with a nonempty `documents` array. Each document requires nonempty string `id`, `title`, `source_url`, `classification` and `text`; an optional object `source_record` carries source metadata. IDs must be unique and safe as identifier segments. URLs must be public-style HTTPS addresses; file URLs, local hosts, literal IP addresses and path traversal are rejected. Questions must be a nonempty list of nonempty strings. This is a local pure-data interface: it does not fetch from a network or read hidden local paths.

The index splits each exact corpus text into successive, nonoverlapping spans of at most 80 whitespace-delimited words, preserving the original character offsets and literal substring for each span. IDs use `<document_id>:chunk:<four-digit ordinal>`. The code fits scikit-learn 1.9.1 `TfidfVectorizer` on those spans, with lowercase, English stop words, word unigrams/bigrams and L2 norm. For each question it saves up to three positive-score results, sorted by descending cosine-equivalent TF-IDF score, then document ID and chunk ID. Scores are rounded to ten decimals. The `0.12` minimum is a predeclared demonstration threshold, **not a calibrated relevance probability**. No indexed term overlap causes `insufficient_evidence` with reason `no_indexed_term_overlap`; a positive top score under the threshold causes `below_score_threshold`.

Before emitting any excerpt, the module also requires every non-stopword query token to occur in that same top chunk. An issuer match by itself therefore cannot turn a request for an absent metric or year into a claimed answer. Missing tokens produce `query_terms_not_in_cited_excerpt` and are logged. This lexical guard is deliberately conservative and may abstain on valid paraphrases; it is not semantic entailment.

When supported, `answer_excerpt` is exactly the top chunk's archived text, and `citation` includes `document_id`, `chunk_id`, SEC `source_url`, character offsets, excerpt, chunk SHA-256, rank and score. There is no generated synthesis or numerical calculation. If unsupported, the answer excerpt and citation are null. Both cases retain the ranked results for review.

`verify_retrieval` reruns the same index and all questions against the supplied frozen corpus, then compares corpus hash, document metadata, ranks, scores, excerpts, citation fields, abstention status and configuration. Mutation of a citation, score, answer or corpus text is detected even if an outer file manifest is rehashed. It does not authenticate the original SEC server, prove the financial meaning of a fact or prevent an actor from replacing the entire corpus and evidence with a new consistent pair.

## Reproduction and observed smoke result

From the repository root, in the Python 3.12 environment after installing the project:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_retrieval.py' -v
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from yk_evidence.retrieval import run_retrieval, verify_retrieval
corpus = json.loads(Path('src/yk_evidence/assets/data/financial/corpus_v04_sec_companyfacts.json').read_text())
questions = ['Microsoft fiscal 2025 revenue', 'Apple fiscal 2025 revenue', 'NVIDIA fiscal 2025 revenue', 'Microsoft employee count fiscal 2025', 'Microsoft revenue fiscal 2035', 'unicorn zeppelin nebula']
evidence = run_retrieval(corpus, questions)
print([(item['status'], item['citation']['document_id'] if item['citation'] else None) for item in evidence['queries']])
print(verify_retrieval(corpus, evidence))
PY
```

The local run on the frozen corpus returned Microsoft, Apple and NVIDIA as the top cited documents for the three named revenue queries. It abstained on Microsoft employee count and fiscal 2035 despite lexical matches on the issuer, and on the unrelated query with no overlap. It returned `[]` verification errors. These six hand-written smoke questions are **not** an independently annotated retrieval benchmark; no recall, answer accuracy or broad financial reasoning claim follows from them.

## Limits and next evidence step

This module makes the provenance and retrieval seam testable; the packaged corpus remains deliberately narrow. Company Facts omits full filing narrative and company-specific extension concepts. A literal excerpt from a local rendering should not be presented as a direct quotation of Form 10-K prose. The TF-IDF score measures lexical similarity, not whether a passage entails an answer. Before any broader financial-analysis claim, freeze real filing HTML, map passages to source offsets, create a reviewer-annotated question set including wrong-period and unanswerable questions, and report predeclared retrieval and citation outcomes from raw runs. No LLM was called in this v0.4 slice.
