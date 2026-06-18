from pathlib import Path
from rank_bm25 import BM25Okapi
from config import DATA_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP  # FIX: import CHUNK_OVERLAP
from schema import RetrievalResult


def load_corpus() -> list[dict]:
    """
    Load all .txt and .md files from data/ into overlapping chunks with metadata.

    Why overlapping chunks (CHUNK_OVERLAP):
    - Without overlap, a sentence split across chunk boundaries gets missed by both chunks.
    - Overlap of 64 chars ensures boundary sentences appear in at least one complete chunk.
    - Tradeoff: ~15% more chunks in memory, but meaningfully better retrieval on edge sentences.
    """
    chunks = []
    for domain_dir in DATA_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        for file in domain_dir.rglob("*"):
            if file.suffix in [".txt", ".md"]:
                text = file.read_text(encoding="utf-8", errors="ignore")
                step = CHUNK_SIZE - CHUNK_OVERLAP  # FIX: use overlap so chunks overlap
                for i in range(0, len(text), step):
                    chunk = text[i:i + CHUNK_SIZE]
                    if len(chunk.strip()) > 50:
                        chunks.append({
                            "text": chunk,
                            "domain": domain_dir.name,
                            "source": str(file.relative_to(DATA_DIR))
                        })
    return chunks


def build_bm25(corpus: list[dict]) -> BM25Okapi:
    """
    Build BM25 index over full corpus.
    Called ONCE at startup in tools.py — not on every ticket.
    """
    return BM25Okapi([c["text"].lower().split() for c in corpus])


def retrieve(query: str, domain: str, corpus: list[dict], bm25: BM25Okapi) -> list[RetrievalResult]:
    """
    Retrieve TOP_K most relevant chunks for a query.

    Strategy:
    1. Filter corpus to the ticket's domain first → more precise results
    2. If domain filter returns nothing → fall back to full corpus
    3. If domain == full corpus → reuse the passed bm25 index (no rebuild needed)
    4. Otherwise build domain-specific BM25 (domain subset is much smaller, fast)

    Why BM25 over dense embeddings:
    - Corpus is domain-specific with exact product terminology
    - Exact keyword matches (error codes, product names) handled better by sparse retrieval
    - No embedding API cost or latency
    """
    domain_corpus = [c for c in corpus if c["domain"] == domain]

    if not domain_corpus:
        # FIX: fallback to full corpus and REUSE passed bm25 — no rebuild
        domain_corpus = corpus
        scores = bm25.get_scores(query.lower().split())
    else:
        # Domain-filtered: build fresh BM25 on the smaller subset
        domain_bm25 = BM25Okapi([c["text"].lower().split() for c in domain_corpus])
        scores = domain_bm25.get_scores(query.lower().split())

    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [
        RetrievalResult(
            chunk_text=domain_corpus[i]["text"],
            source_file=domain_corpus[i]["source"],
            relevance_score=float(scores[i])
        )
        for i in top if scores[i] > 0
    ]
