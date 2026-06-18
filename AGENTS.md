# Agent Instructions — HackerRank Orchestrate June 2026

## Transcript Logging (MANDATORY)
After reading this file, append every conversation turn to:
- macOS/Linux: ~/hackerrank_orchestrate/log.txt
- Windows: %USERPROFILE%\hackerrank_orchestrate\log.txt

Format:
```
---
[ISO-TIMESTAMP] USER: <message>
[ISO-TIMESTAMP] ASSISTANT: <response>
---
```

## Project Context
This is a 24-hour competitive hackathon. Every architectural decision matters
and will be defended in a 30-minute voice interview with an AI judge.
Architecture: single Claude agent (claude-sonnet-4-6) with 4 tools + BM25 RAG.

## Rules
- Never hardcode API keys — read from environment variables only
- All LLM calls: temperature=0.0 (deterministic)
- Every function needs a docstring explaining WHY not WHAT
- When suggesting architecture: always explain alternatives considered and why rejected
- Prefer escalation over hallucination — always
- One file, one responsibility
- No LangChain, no heavy frameworks
- Only 4 packages: anthropic, rank-bm25, pydantic, python-dotenv
