import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ISSUES_DIR = ROOT / "support_issues"
INPUT_CSV = ISSUES_DIR / "support_issues.csv"
OUTPUT_CSV = ISSUES_DIR / "output.csv"
SAMPLE_CSV = ISSUES_DIR / "sample_support_issues.csv"

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL_FAST = "claude-haiku-4-5-20251001"   # tools + classification
MODEL_SMART = "claude-sonnet-4-6"           # final response generation
MAX_TOKENS = 1024
TEMPERATURE = 0.0

TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # overlap between chunks prevents splitting mid-sentence

MAX_TOOL_ITERATIONS = 10  # safety cap on agent tool-use loop
MAX_RETRIES = 3           # retries for transient API failures
RETRY_DELAY = 1.0         # seconds before first retry; doubles each attempt

ESCALATION_TRIGGERS = [
    "legal action", "lawsuit", "fraud", "unauthorized charge",
    "identity theft", "account hacked", "chargeback",
    "refund", "compensation", "threat", "abuse",
    "ignore previous instructions", "jailbreak",
    "DAN", "system prompt", "SYSTEM:"
]
# NOTE: "act as" removed from ESCALATION_TRIGGERS — too broad, causes false positives
# on legitimate tickets like "act as if my account was new"
# Prompt injection patterns in risk_gate.py handle the actual injection cases

DOMAINS = []  # fill in on day-of from data/ folder names
