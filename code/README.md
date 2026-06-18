# HackerRank Orchestrate June 2026 — Agent

## Setup

```bash
cd code
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Run on sample data (test first)

```bash
cd code
python main.py --sample
```

Output written to: `support_issues/output.csv`

## Run on full dataset

```bash
cd code
python main.py
```

## Override paths

```bash
python main.py --input /path/to/input.csv --output /path/to/output.csv
```

## Architecture

Single Claude agent (claude-sonnet-4-6) with 4 tools:

```
ticket_in
    ↓
[risk_gate]          ← keyword-based, NO LLM, instant
    ↓ (if passes)
[Claude Agent]
    ↓ calls:
    ├── classify_ticket(text) → {domain, request_type}
    ├── retrieve_docs(query, domain) → corpus chunks via BM25
    └── check_escalation(ticket, chunks) → {should_escalate, reason}
    ↓
[Pydantic AgentOutput] → output CSV row (flushed immediately)
```

## Dependencies

- `anthropic` — Claude API
- `rank-bm25` — BM25 retrieval over provided corpus
- `pydantic` — output schema validation
- `python-dotenv` — environment variable loading

No LangChain. No vector DB. No external web calls.

## Output CSV columns

| Column | Values |
|---|---|
| ticket_id | from input |
| status | replied / escalated |
| product_area | domain name |
| response | agent response or empty string if escalated |
| justification | specific reason (never generic) |
| request_type | product_issue / feature_request / bug / invalid |
