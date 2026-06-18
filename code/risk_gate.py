from config import ESCALATION_TRIGGERS


def should_escalate(text: str) -> tuple[bool, str]:
    """
    Pre-retrieval safety gate. Runs BEFORE any LLM call.
    Purpose: catch prompt injection and high-risk tickets instantly without spending tokens.

    Why keyword-based and not LLM-based:
    - Zero latency — no API call needed
    - Deterministic — same input always same output
    - Prompt injection must be caught BEFORE the LLM sees the text

    Returns (should_escalate: bool, reason: str)
    """
    t = text.lower()

    # Prompt injection / jailbreak detection
    # NOTE: patterns are specific enough to avoid false positives on legitimate tickets.
    # "act as" removed — too broad (e.g. "act as if my account was reset" is legitimate).
    # Using more specific multi-word patterns instead.
    injection_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "new instructions:",
        "you are now",
        "act as if you are an ai",
        "act as an ai",
        "jailbreak",
        "dan mode",
        "\nsystem:",       # newline before system: catches injected system blocks
        "[system]",
        "disregard your instructions",
    ]
    for pattern in injection_patterns:
        if pattern in t:
            return True, f"Prompt injection detected: '{pattern}'"

    # High-risk keywords requiring human escalation
    for trigger in ESCALATION_TRIGGERS:
        if trigger.lower() in t:
            return True, f"High-risk keyword: '{trigger}'"

    return False, ""
