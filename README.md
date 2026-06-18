# HackerRank Orchestrate June 2026

24-hour solo hackathon submission by Ritesh

## What It Does
Terminal AI agent that reads support tickets from a CSV, classifies them by domain,
retrieves relevant documentation via BM25 RAG, and either replies with a grounded
answer or escalates. Outputs one row per ticket to output.csv.

## Architecture
Single Claude agent (claude-sonnet-4-6) with 4 tools + BM25 RAG.
Pre-retrieval risk gate catches prompt injection before spending any tokens.

See `code/README.md` for setup and run instructions.

## Evaluation Criteria
See `evalutation_criteria.md` (copied from official May edition repo).
