import json
from anthropic import Anthropic
from retriever import load_corpus, build_bm25, retrieve
from config import MODEL_FAST, MAX_TOKENS, TEMPERATURE, DOMAINS, TOP_K

client = Anthropic()
CORPUS = load_corpus()
BM25 = build_bm25(CORPUS)


def retrieve_docs(query: str, domain: str) -> str:
    """
    Multi-query BM25 retrieval. Runs 3 query variations, merges by source.
    Why 3 queries: BM25 is term-sensitive. User words often differ from corpus words.
    3 variations catches paraphrased tickets without any embedding cost.
    """
    try:
        rephrase_response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Produce 2 alternative phrasings of this query using different words, same meaning.
Query: {query}
Output ONLY valid JSON: {{"q1": "...", "q2": "..."}}"""}]
        )
        alt = json.loads(rephrase_response.content[0].text)
        queries = [query, alt.get("q1", query), alt.get("q2", query)]
    except Exception:
        queries = [query]

    seen_sources = set()
    all_results = []
    for q in queries:
        for r in retrieve(q, domain, CORPUS, BM25):
            if r.source_file not in seen_sources:
                seen_sources.add(r.source_file)
                all_results.append(r)

    all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    final = all_results[:TOP_K]

    if not final:
        return "No relevant documentation found."
    return "\n---\n".join([f"[{r.source_file}]\n{r.chunk_text}" for r in final])


def classify_ticket(text: str) -> str:
    """
    Classify domain + request_type using Haiku (fast + cheap).
    Why Haiku: classification is routing, not reasoning. 10x cheaper than Sonnet.
    Error fallback returns safe JSON so agent loop never crashes on bad tool result.
    """
    domain_list = ", ".join(DOMAINS) if DOMAINS else "unknown"
    try:
        response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Classify this support ticket.

Ticket: {text}

Available domains: {domain_list}
Request types: product_issue, feature_request, bug, invalid

Respond with ONLY valid JSON: {{"domain": "...", "request_type": "..."}}"""}]
        )
        return response.content[0].text
    except Exception as e:
        return json.dumps({"domain": "unknown", "request_type": "product_issue"})


def check_escalation(ticket_text: str, retrieved_chunks: str) -> str:
    """
    Decide if ticket needs escalation given retrieved docs.
    Philosophy: false escalations are cheap, hallucinated policies are expensive.
    Error fallback escalates for safety — never hallucinate on API failure.
    """
    try:
        response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": f"""Should this ticket be escalated?

Ticket: {ticket_text}

Available documentation:
{retrieved_chunks}

Escalate if: docs don't cover the issue, involves fraud/legal/security, or confidence is low.
Do NOT escalate if docs clearly answer the question.

Respond with ONLY valid JSON: {{"should_escalate": true, "reason": "specific reason"}}"""}]
        )
        return response.content[0].text
    except Exception as e:
        return json.dumps({"should_escalate": True, "reason": f"Escalation check failed: {str(e)}"})
