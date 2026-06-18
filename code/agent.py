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

SYSTEM_PROMPT = """You are a precise support triage agent. Process each ticket in this exact order:
1. classify_ticket → get domain and request_type
2. retrieve_docs → get relevant corpus chunks (use domain from step 1)
3. check_escalation → decide if escalation needed
4. format_output → structure the final answer (ALWAYS call this last)

Rules:
- NEVER make up policies not in the retrieved docs
- If docs don't cover the issue → escalate
- justification must be specific: name the exact corpus gap or risk reason
- If escalating: response must be empty string"""


def _call_tool(name: str, inputs: dict) -> str:
    if name == "retrieve_docs":
        return retrieve_docs(inputs["query"], inputs["domain"])
    elif name == "classify_ticket":
        return classify_ticket(inputs["text"])
    elif name == "check_escalation":
        return check_escalation(inputs["ticket_text"], inputs["retrieved_chunks"])
    elif name == "format_output":
        return json.dumps(inputs)  # just echo back — we capture this below
    return f"Unknown tool: {name}"


def run_agent(raw_ticket: dict) -> AgentOutput:
    """
    Single Claude agent with 4 tools. format_output is forced as final step.
    Why forced tool_choice for output: eliminates regex JSON parsing entirely.
    Why single agent over multi-agent: official data shows single agent won (rank 1),
    multi-agent averaged rank 468.
    """
    ticket_id = raw_ticket.get("ticket_id", "unknown")
    message = raw_ticket.get("message", raw_ticket.get("body", ""))

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

    for _ in range(MAX_TOOL_ITERATIONS):
        # For the final step, force format_output to guarantee schema compliance
        is_final_step = len(messages) >= 7  # classify + retrieve + escalation done
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
            tools=TOOLS,
            tool_choice=tool_choice,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Should not happen if format_output forced correctly — escalate for safety
            return AgentOutput(
                ticket_id=ticket_id, status="escalated", product_area="unknown",
                response="", justification="Agent ended without calling format_output",
                request_type="product_issue"
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "format_output":
                        # Capture structured output directly from tool inputs
                        try:
                            data = dict(block.input)
                            data["ticket_id"] = ticket_id
                            return AgentOutput(**data)
                        except Exception as e:
                            return AgentOutput(
                                ticket_id=ticket_id, status="escalated",
                                product_area="unknown", response="",
                                justification=f"format_output schema error: {e}",
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
                response="", justification=f"Unexpected stop: {response.stop_reason}",
                request_type="product_issue"
            )

    return AgentOutput(
        ticket_id=ticket_id, status="escalated", product_area="unknown",
        response="", justification=f"Exceeded {MAX_TOOL_ITERATIONS} iterations",
        request_type="product_issue"
    )
