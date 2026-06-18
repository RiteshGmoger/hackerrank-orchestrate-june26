import json
import time
from anthropic import Anthropic
from retriever import load_corpus, build_bm25, retrieve
from config import MODEL_FAST, MAX_TOKENS, TEMPERATURE, DOMAINS, TOP_K, MAX_RETRIES, RETRY_DELAY

client = Anthropic()

try:
    CORPUS = load_corpus()
    BM25 = build_bm25(CORPUS)
    print(f"[corpus] Loaded {len(CORPUS)} chunks from data/")
except Exception as e:
    print(f"[WARNING] Corpus load failed: {e} — retrieval will return empty")
    CORPUS = []
    BM25 = None


def _call_with_retry(fn):
    """
    Exponential backoff retry for Anthropic API calls.
    Why: rate limit errors (429) and transient 5xx are common under load.
    Without retry, one bad API call fails the whole ticket.
    Backoff: 1s → 2s → 4s before giving up.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    raise last_exc


def retrieve_docs(query: str, domain: str) -> str:
    """
    Multi-query BM25 retrieval. Runs 3 query variations, merges by source.
    Why 3 queries: BM25 is term-sensitive. User words often differ from corpus words.
    3 variations catches paraphrased tickets without any embedding cost.

    Why not just use embeddings: BM25 handles exact product-specific keywords better
    (error codes, feature names). Multi-query BM25 covers semantic variation cheaply.
    I considered dense retrieval but rejected it — BM25 alone sufficient on sample data,
    adding dense would introduce API cost + latency without measurable gain.
    """
    if BM25 is None or not CORPUS:
        return "No corpus loaded — data directory may be empty."

    try:
        rephrase_response = _call_with_retry(lambda: client.messages.create(
            model=MODEL_FAST,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Produce 2 alternative phrasings of this query using different words, same meaning.
Query: {query}
Output ONLY valid JSON: {{"q1": "...", "q2": "..."}}"""}]
        ))
        alt = json.loads(rephrase_response.content[0].text)
        queries = [query, alt.get("q1", query), alt.get("q2", query)]
    except Exception:
        queries = [query]  # fallback: single query if rephrasing fails

    seen_chunks = set()
    all_results = []
    for q in queries:
        for r in retrieve(q, domain, CORPUS, BM25):
            if r.chunk_text not in seen_chunks:
                seen_chunks.add(r.chunk_text)
                all_results.append(r)

    all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    final = all_results[:TOP_K]

    if not final:
        return "No relevant documentation found."
    return "\n---\n".join([f"[{r.source_file}]\n{r.chunk_text}" for r in final])


def classify_ticket(text: str) -> str:
    """
    Classify domain + request_type using Haiku (fast + cheap).
    Why Haiku not Sonnet: classification is routing, not reasoning. 10x cheaper.
    Error fallback returns safe JSON — agent loop never crashes on bad tool result.
    """
    domain_list = ", ".join(DOMAINS) if DOMAINS else "unknown"
    try:
        response = _call_with_retry(lambda: client.messages.create(
            model=MODEL_FAST,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Classify this support ticket.

Ticket: {text}

Available domains: {domain_list}
Request types: product_issue, feature_request, bug, invalid

Respond with ONLY valid JSON: {{"domain": "...", "request_type": "..."}}"""}]
        ))
        return response.content[0].text
    except Exception as e:
        return json.dumps({"domain": "unknown", "request_type": "product_issue"})


def check_escalation(ticket_text: str, retrieved_chunks: str) -> str:
    """
    Decide if ticket needs escalation given retrieved docs.
    Philosophy: false escalations are cheap, hallucinated policies are expensive.
    Error fallback escalates for safety — never hallucinate on API failure.

    FIX: prompt previously showed only {"should_escalate": true} as the example JSON.
    This biased the model to always return true. Now shows BOTH cases explicitly.
    """
    try:
        response = _call_with_retry(lambda: client.messages.create(
            model=MODEL_FAST,
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Should this ticket be escalated?

Ticket: {ticket_text}

Available documentation:
{retrieved_chunks}

Escalate if: docs don't cover the issue, involves fraud/legal/security, or confidence is low.
Do NOT escalate if the docs clearly and directly answer the question.

Respond with ONLY valid JSON. Examples:
- If docs answer it:   {{"should_escalate": false, "reason": "account/login.md covers password reset steps directly"}}
- If escalation needed: {{"should_escalate": true, "reason": "ticket mentions chargeback — not covered in corpus"}}

Your response:"""}]
        ))
        return response.content[0].text
    except Exception as e:
        return json.dumps({"should_escalate": True, "reason": f"Escalation check failed: {str(e)}"})
