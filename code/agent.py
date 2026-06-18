import json
from anthropic import Anthropic
from config import MODEL_SMART, MAX_TOKENS, TEMPERATURE, MAX_TOOL_ITERATIONS
from risk_gate import should_escalate
from tools import retrieve_docs, classify_ticket, check_escalation
from schema import AgentOutput

client = Anthropic()

TOOLS = [
    {
        "name": "classify_ticket",
        "description": "Classify a support ticket's domain and request type. Call this FIRST.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "retrieve_docs",
        "description": "Retrieve relevant documentation chunks via BM25. Call AFTER classify_ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "domain": {"type": "string"}
            },
            "required": ["query", "domain"]
        }
    },
    {
        "name": "check_escalation",
        "description": "Determine if ticket should be escalated. Call AFTER retrieve_docs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_text": {"type": "string"},
                "retrieved_chunks": {"type": "string"}
            },
            "required": ["ticket_text", "retrieved_chunks"]
        }
    },
    {
        "name": "format_output",
        "description": "Format the final structured output. Call this LAST after all other tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "status": {"type": "string", "enum": ["replied", "escalated"]},
                "product_area": {"type": "string"},
                "response": {"type": "string", "description": "Empty string if escalated"},
                "justification": {"type": "string", "description": "Specific reason — never generic"},
                "request_type": {
                    "type": "string",
                    "enum": ["product_issue", "feature_request", "bug", "invalid"]
                }
            },
            "required": ["ticket_id", "status", "product_area", "response", "justification", "request_type"]
        }
    }
]

# FIX: two separate tool lists.
# TOOLS_PROCESS: only the 3 working tools — format_output is NOT visible to the model yet.
# Without this split, Claude can call format_output prematurely (before classify/retrieve/check),
# producing an incomplete output that bypasses all retrieval.
# TOOLS_ALL: all 4 tools, only exposed when all 3 required tools have been called.
TOOLS_PROCESS = TOOLS[:3]   # classify_ticket, retrieve_docs, check_escalation
TOOLS_ALL = TOOLS            # includes format_output — exposed only at final step

REQUIRED_BEFORE_OUTPUT = {"classify_ticket", "retrieve_docs", "check_escalation"}

SYSTEM_PROMPT = """You are a precise support triage agent. Process each ticket in this exact order:
1. classify_ticket → get domain and request_type
2. retrieve_docs → get relevant corpus chunks (use domain from step 1)
3. check_escalation → decide if escalation needed
4. format_output → structure the final answer (ALWAYS call this last)

Rules:
- NEVER make up policies not in the retrieved docs
- If docs don't cover the issue → escalate
- If escalating: response must be empty string ""

Justification field MUST be specific — generic justifications score 0.
Format: "[action] because [exact reason]. Reference: [source file if applicable]"

BAD:  "Escalated: out of scope"
GOOD: "Escalated: ticket requests chargeback on order placed 45 days ago — corpus covers disputes within 30 days only (billing/refund_policy.md)"

BAD:  "Replied with policy information"
GOOD: "Replied: password reset steps found in account/authentication.md — provided 4-step flow matching user platform (web)"

BAD:  "High-risk keyword detected"
GOOD: "Escalated by risk gate: message contains 'unauthorized charge' — financial fraud pattern requires human review, not automated response" """


def _call_tool(name: str, inputs: dict) -> str:
    if name == "retrieve_docs":
        return retrieve_docs(inputs["query"], inputs["domain"])
    elif name == "classify_ticket":
        return classify_ticket(inputs["text"])
    elif name == "check_escalation":
        return check_escalation(inputs["ticket_text"], inputs["retrieved_chunks"])
    elif name == "format_output":
        return json.dumps(inputs)  # echo back — captured below in the caller
    return f"Unknown tool: {name}"


def run_agent(raw_ticket: dict) -> AgentOutput:
    """
    Single Claude agent with 4 tools. format_output forced as final step.

    Why forced tool_choice for output: eliminates regex JSON parsing entirely.
    format_output schema IS the schema — validation happens in Pydantic, not string parsing.

    Why single agent over multi-agent: official post-mortem data shows single agent
    with tools won (rank 1), multi-agent averaged rank 468.

    Why two tool lists (TOOLS_PROCESS vs TOOLS_ALL): prevents model from calling
    format_output before retrieve/classify/check have run. Without this, early
    format_output calls bypass all retrieval and produce hallucinated outputs.
    """
    ticket_id = raw_ticket.get("ticket_id", "unknown")
    message = raw_ticket.get("message", raw_ticket.get("body", ""))

    # Pre-check: no LLM needed, instant, catches injection + high-risk keywords
    escalate, reason = should_escalate(message)
    if escalate:
        return AgentOutput(
            ticket_id=ticket_id,
            status="escalated",
            product_area="unknown",
            response="",
            justification=f"Pre-escalated by risk gate: {reason}",
            request_type="product_issue"
        )

    messages = [{
        "role": "user",
        "content": (
            f"Process this ticket:\n\n"
            f"Ticket ID: {ticket_id}\n"
            f"Message: {message}\n\n"
            f"Use tools in order: classify_ticket → retrieve_docs → check_escalation → format_output"
        )
    }]

    # Track which tools have been called — robust vs naive message count.
    # Message count breaks if agent calls a tool twice or changes order.
    tools_called: set[str] = set()

    for _ in range(MAX_TOOL_ITERATIONS):
        is_final_step = REQUIRED_BEFORE_OUTPUT.issubset(tools_called)

        # FIX: only expose format_output when all 3 required tools have been called.
        # During processing steps, model only sees the 3 working tools — it cannot
        # call format_output early even if it "wants" to.
        current_tools = TOOLS_ALL if is_final_step else TOOLS_PROCESS
        tool_choice = (
            {"type": "tool", "name": "format_output"}
            if is_final_step
            else {"type": "auto"}
        )

        response = client.messages.create(
            model=MODEL_SMART,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=current_tools,
            tool_choice=tool_choice,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Should not happen with forced format_output — escalate for safety
            return AgentOutput(
                ticket_id=ticket_id, status="escalated", product_area="unknown",
                response="", justification="Agent ended without calling format_output — escalating for safety",
                request_type="product_issue"
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tools_called.add(block.name)

                if block.name == "format_output":
                    # Capture structured output directly from tool inputs.
                    # Pydantic validates — if schema error, escalate with specific reason.
                    try:
                        data = dict(block.input)
                        data["ticket_id"] = ticket_id  # always override — model sometimes fills wrong id
                        return AgentOutput(**data)
                    except Exception as e:
                        return AgentOutput(
                            ticket_id=ticket_id, status="escalated",
                            product_area="unknown", response="",
                            justification=f"format_output schema validation failed: {e}",
                            request_type="product_issue"
                        )

                result = _call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        else:
            return AgentOutput(
                ticket_id=ticket_id, status="escalated", product_area="unknown",
                response="", justification=f"Unexpected stop reason '{response.stop_reason}' — escalating for safety",
                request_type="product_issue"
            )

    return AgentOutput(
        ticket_id=ticket_id, status="escalated", product_area="unknown",
        response="", justification=f"Exceeded {MAX_TOOL_ITERATIONS} tool iterations without producing output",
        request_type="product_issue"
    )
