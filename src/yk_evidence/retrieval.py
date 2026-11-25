"""Deterministic, extractive retrieval over a frozen, source-declared corpus.

The returned excerpt is a literal span from the archived corpus. This module
does not generate an answer, download sources, or assert that a source fact is
true: those are separate provenance and review responsibilities.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from urllib.parse import unquote, urlsplit

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


SCHEMA_VERSION = "0.4"
METHOD = "tfidf_extractive"
CHUNK_WORDS = 80
TOP_K = 3
MIN_SCORE = 0.12
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("corpus must be finite JSON data") from exc
    return _sha256(payload)


def _validate_source_url(source_url: str) -> None:
    try:
        parsed = urlsplit(source_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid source_url") from exc
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise ValueError("source_url must be a public HTTPS URL without credentials")
    if port not in (None, 443) or parsed.fragment:
        raise ValueError("source_url has an unsafe port or fragment")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("source_url cannot address a local host")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("source_url cannot use a literal IP address")
    decoded_path = unquote(parsed.path)
    if (
        "\\" in source_url
        or "%" in decoded_path
        or any(part in (".", "..") for part in decoded_path.split("/"))
    ):
        raise ValueError("source_url has an unsafe local path")


def _validate_corpus(corpus: dict) -> list[dict]:
    if not isinstance(corpus, dict) or corpus.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"corpus schema_version must be {SCHEMA_VERSION}")
    documents = corpus.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("corpus documents must be a nonempty list")
    ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"document {index} must be an object")
        for key in ("id", "title", "source_url", "classification", "text"):
            if not isinstance(document.get(key), str) or not document[key].strip():
                raise ValueError(f"document {index} requires nonempty {key}")
        document_id = document["id"]
        if not _ID_PATTERN.fullmatch(document_id) or ".." in document_id:
            raise ValueError(f"invalid document id: {document_id!r}")
        if document_id in ids:
            raise ValueError(f"duplicate document id: {document_id}")
        ids.add(document_id)
        _validate_source_url(document["source_url"])
        if "\x00" in document["text"]:
            raise ValueError(f"document {document_id} text contains a NUL byte")
        if "source_record" in document and not isinstance(document["source_record"], dict):
            raise ValueError(f"document {document_id} source_record must be an object")
    _json_sha256(corpus)
    return documents


def _chunks(documents: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        text = document["text"]
        words = list(re.finditer(r"\S+", text))
        for number, first in enumerate(range(0, len(words), CHUNK_WORDS)):
            start = words[first].start()
            end = words[min(first + CHUNK_WORDS, len(words)) - 1].end()
            excerpt = text[start:end]
            chunks.append(
                {
                    "document_id": document["id"],
                    "chunk_id": f"{document['id']}:chunk:{number:04d}",
                    "source_url": document["source_url"],
                    "start_char": start,
                    "end_char": end,
                    "excerpt": excerpt,
                    "chunk_sha256": _sha256(excerpt.encode("utf-8")),
                }
            )
    return chunks


def run_retrieval(corpus: dict, questions: list[str]) -> dict:
    """Rank literal corpus chunks with TF-IDF and cite or abstain per question.

    The corpus has schema_version ``0.4`` and a nonempty ``documents`` list.
    Every document must declare ``id``, ``title``, HTTPS ``source_url``,
    ``classification`` and ``text``. ``questions`` is a nonempty list of
    nonempty strings. Invalid inputs raise ``ValueError``. No network I/O is
    performed; callers must archive the same corpus alongside the evidence.
    """
    documents = _validate_corpus(corpus)
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("each question must be a nonempty string")

    chunks = _chunks(documents)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
    )
    try:
        chunk_matrix = vectorizer.fit_transform([chunk["excerpt"] for chunk in chunks])
    except ValueError as exc:
        raise ValueError("corpus has no indexable terms") from exc

    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    stop_words = vectorizer.get_stop_words() or set()

    def substantive_terms(value: str) -> set[str]:
        return set(tokenize(preprocess(value))) - stop_words

    results = []
    for question in questions:
        query_vector = vectorizer.transform([question])
        scores = (chunk_matrix @ query_vector.T).toarray().reshape(-1)
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-float(scores[index]), chunks[index]["document_id"], chunks[index]["chunk_id"]),
        )[:TOP_K]
        ranked_results = [
            {"rank": rank, "score": round(float(scores[index]), 10), **chunks[index]}
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        top = ranked_results[0] if ranked_results else None
        unmatched_terms = sorted(substantive_terms(question) - substantive_terms(top["excerpt"])) if top else []
        if top and top["score"] >= MIN_SCORE and not unmatched_terms:
            results.append(
                {
                    "question": question,
                    "status": "supported",
                    "reason": None,
                    "answer_type": "verbatim_excerpt",
                    "answer_excerpt": top["excerpt"],
                    "citation": top,
                    "unmatched_query_terms": [],
                    "ranked_results": ranked_results,
                }
            )
        else:
            if not top:
                reason = "no_indexed_term_overlap"
            elif top["score"] < MIN_SCORE:
                reason = "below_score_threshold"
            else:
                reason = "query_terms_not_in_cited_excerpt"
            results.append(
                {
                    "question": question,
                    "status": "insufficient_evidence",
                    "reason": reason,
                    "answer_type": "abstention",
                    "answer_excerpt": None,
                    "citation": None,
                    "unmatched_query_terms": unmatched_terms,
                    "ranked_results": ranked_results,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "corpus_sha256": _json_sha256(corpus),
        "config": {
            "chunk_words": CHUNK_WORDS,
            "overlap_words": 0,
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "vectorizer": "TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), token_pattern=r'(?u)\\b\\w+\\b', norm='l2')",
            "scikit_learn_version": sklearn.__version__,
        },
        "documents": [
            {
                "id": document["id"],
                "title": document["title"],
                "source_url": document["source_url"],
                "classification": document["classification"],
                "text_sha256": _sha256(document["text"].encode("utf-8")),
                "chunk_count": sum(chunk["document_id"] == document["id"] for chunk in chunks),
                **({"source_record": document["source_record"]} if "source_record" in document else {}),
            }
            for document in documents
        ],
        "queries": results,
        "limitations": (
            "Extractive TF-IDF retrieval of literal archived text only; no LLM generation, "
            "semantic entailment, filing-level verification, or investment advice."
        ),
    }


def _differences(expected: object, actual: object, path: str, errors: list[str]) -> None:
    if len(errors) >= 25:
        return
    if type(expected) is not type(actual):
        errors.append(f"{path}: type differs")
    elif isinstance(expected, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child_path = f"{path}.{key}"
            if key not in expected:
                errors.append(f"{child_path}: unexpected field")
            elif key not in actual:
                errors.append(f"{child_path}: missing field")
            else:
                _differences(expected[key], actual[key], child_path, errors)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: item count differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _differences(left, right, f"{path}[{index}]", errors)
    elif expected != actual:
        errors.append(f"{path}: value differs")


def verify_retrieval(corpus: dict, evidence: dict) -> list[str]:
    """Recompute rankings, scores and citations from the frozen corpus.

    Returns an empty list for exact semantic agreement, or a list of paths
    that differ. This is an integrity/consistency check, not authentication of
    the SEC's live site or a signature over the archived corpus.
    """
    if not isinstance(evidence, dict):
        return ["evidence must be an object"]
    queries = evidence.get("queries")
    if not isinstance(queries, list) or not queries:
        return ["evidence.queries must be a nonempty list"]
    if any(not isinstance(item, dict) or not isinstance(item.get("question"), str) for item in queries):
        return ["evidence.queries contains an invalid question"]
    try:
        expected = run_retrieval(corpus, [item["question"] for item in queries])
    except ValueError as exc:
        return [f"cannot recompute retrieval: {exc}"]
    errors: list[str] = []
    _differences(expected, evidence, "evidence", errors)
    return errors
