"""Hide secret-shaped strings in tool output before the model sees them.

The path boundary blocks secret *files* (.env, *.pem). This catches secret
*content* that sits in ordinary files: a key pasted into a test, a config, a
notebook. It runs on every tool's output in one place (the agent loop), so a
new tool can't forget it.

Pattern matching can't catch every secret (false negatives) and will sometimes
hide harmless text (false positives). It's a safety net, not a vault.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic/openai-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL)),
    # key = "value" style assignments, e.g. API_KEY="abc123...", password: hunter2hunter2
    ("assigned secret", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:api[_-]?key|secret|token|password|passwd))\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{8,}\3")),
]


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0
    for name, pattern in PATTERNS:
        if name == "assigned secret":
            text, n = pattern.subn(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
        else:
            text, n = pattern.subn(REDACTED, text)
        count += n
    return text, count
