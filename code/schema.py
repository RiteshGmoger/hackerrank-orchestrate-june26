from pydantic import BaseModel
from typing import Literal


class TicketInput(BaseModel):
    """
    Normalized input ticket.
    product field is optional — not all CSVs will have it.
    agent.py uses raw dict .get() so this is a reference schema, not enforced on input.
    """
    ticket_id: str
    product: str = ""   # default empty — not all input CSVs have this column
    message: str


class AgentOutput(BaseModel):
    """
    Enforced output schema. Every field required.
    justification must never be empty — generic justifications capped at 70/100 by judges.
    """
    ticket_id: str
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


class RetrievalResult(BaseModel):
    """Single BM25 retrieval result with source traceability."""
    chunk_text: str
    source_file: str
    relevance_score: float
