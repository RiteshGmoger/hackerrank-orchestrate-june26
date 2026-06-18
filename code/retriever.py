from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi
from config import DATA_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from schema import RetrievalResult


def load_corpus() -> list[dict]:
    """
    Load all .txt and .md files from data/ into overlapping chunks with metadata.

    Why overlapping chunks (CHUNK_OVERLAP=64): prevents answer-critical sentences
    from being split at chunk boundaries. Last 64 chars of chunk N are the first
    64 chars of chunk N+1 — context at boundaries is preserved.

    Why include domain + source: enables domain-filtered retrieval and
    specific justification references like "billing/refund_policy.md".

    Returns empty list if DATA_DIR doesn't exist — safe before day-of data arrives.
    """
    chunks = []
    if not DATA_DIR.exists():
        return chunks

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)  # stride between chunk starts

    for domain_dir in sorted(DATA_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        for file in sorted(domain_dir.rglob("*")):
            if file.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i in range(0, max(1, len(text) - CHUNK_OVERLAP), step):
                chunk = text[i:i + CHUNK_SIZE]
                if len(chunk.strip()) > 50:  # skip near-empty chunks
                    chunks.append({
                        "text": chunk,
                        "domain": domain,
                        "source": str(file.relative_to(DATA_DIR))
                    })

    return chunks


def build_bm25(corpus: list[dict]) -> Optional[BM25Okapi]:
    """
    Build BM25 index over entire corpus.
    Returns None if corpus is empty — BM25Okapi([]) raises internally.

    Why BM25 over vector/dense retrieval:
    - Corpus is domain-specific with exact product terminology (error codes, feature names)
    - BM25 handles exact keyword match better than dense-only for this use case
    - No embedding API cost or latency
    - I considered adding a dense reranker but rejected it: BM25 alone was sufficient
      on sample data and adding dense retrieval would introduce latency + complexity
      without measurable accuracy gain in testing.
    """
    if not corpus:
        return None
    return BM25Okapi([c["text"].lower().split() for c in corpus])


def retrieve(
    query: str,
    domain: str,
    corpus: list[dict],
    bm25: Optional[BM25Okapi]
) -> list[RetrievalResult]:
    """
    Domain-filtered BM25 retrieval with full-corpus fallback.

    Strategy: filter by domain first for precision. If domain returns no results
    (unknown domain or classifier error), fall back to full corpus — avoids
    empty-handed escalation on tickets that COULD be answered.

    Why rebuild BM25 on the filtered set (not reuse the global bm25):
    BM25 scores are relative to the index they were computed against.
    Reusing global bm25 scores for a domain-filtered subset gives wrong rankings
    because IDF values are computed over all documents.
    """
    if bm25 is None or not corpus:
        return []

    # Domain-filtered subset for precision
    domain_corpus = [c for c in corpus if c["domain"] == domain]
    if not domain_corpus:
        domain_corpus = corpus  # fallback: search full corpus on unknown domain

    # Rebuild BM25 on filtered set — scores relative to subset, not global
    domain_bm25 = BM25Okapi([c["text"].lower().split() for c in domain_corpus])
    scores = domain_bm25.get_scores(query.lower().split())

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]

    return [
        RetrievalResult(
            chunk_text=domain_corpus[i]["text"],
            source_file=domain_corpus[i]["source"],
            relevance_score=float(scores[i])
        )
        for i in top_indices
        if scores[i] > 0
    ]
