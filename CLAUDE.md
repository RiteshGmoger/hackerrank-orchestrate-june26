# Claude Code Context — HackerRank Orchestrate June 2026

## What This Project Is
Terminal AI agent for a 24-hour hackathon.
Reads support tickets from CSV → classifies → retrieves docs → replies or escalates → writes output CSV.

## Architecture (decided, do not change)
Single Claude agent (claude-sonnet-4-6) with 4 tools:
- retrieve_docs: BM25 search over provided corpus
- classify_ticket: domain + request_type classification (Haiku)
- check_escalation: grounding check (Haiku)
- format_output: forced structured output — zero parse errors

Pre-retrieval risk gate (no LLM) → agent tool loop → Pydantic output validation.

## Hard Constraints (never violate these)
- No LangChain, no vector DBs, no heavy frameworks
- Only 4 packages: anthropic, rank-bm25, pydantic, python-dotenv
- temperature=0.0 everywhere — deterministic outputs only
- Never make web calls — search provided corpus ONLY
- Escalate when uncertain, never hallucinate
- justification field must be specific, never generic
- Every function needs a docstring explaining WHY not WHAT

## File Responsibilities
- config.py — all constants, model names, paths, thresholds
- schema.py — Pydantic models for input + output
- risk_gate.py — keyword-based pre-check, no LLM
- retriever.py — BM25 corpus loader + query
- tools.py — 3 tool functions used by agent
- agent.py — single Claude agent with tool-use loop
- main.py — entry point, CSV I/O

## On Day-Of (June 19, 11 AM IST)
1. Update DOMAINS in config.py with actual folder names from data/
2. Update schema.py if output CSV columns differ from template
3. Run: python main.py --sample
4. Everything else is pre-written
