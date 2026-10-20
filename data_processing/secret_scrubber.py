"""
PII & Secret Sanitization Engine
Identifies and redacts API tokens, private encryption keys, IP addresses,
and email addresses to prevent the model from leaking sensitive credentials.
"""

import re
from typing import Tuple, Dict, Any, List

# Compiled regex patterns for common API keys and cryptographic secrets
SECRET_PATTERNS = [
    # AWS Access Keys
    (re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"), "<REDACTED_AWS_KEY>"),
    # GitHub Personal Access Tokens
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b"), "<REDACTED_GITHUB_TOKEN>"),
    # OpenAI API Keys
    (re.compile(r"\bsk-[a-zA-Z0-9]{32,64}\b"), "<REDACTED_OPENAI_KEY>"),
    # Slack Bot & User Tokens
    (re.compile(r"\bxox[baprs]-[0-9a-zA-Z]{10,48}\b"), "<REDACTED_SLACK_TOKEN>"),
    # Generic High-Entropy API Keys (e.g., api_key = "...")
    (
        re.compile(
            r"""(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*["']([A-Za-z0-9/+=_-]{20,80})["']"""
        ),
        r'\1 = "<REDACTED_GENERIC_KEY>"',
    ),
    # RSA / OpenSSH Private Keys
    (
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        "<REDACTED_PRIVATE_KEY>",
    ),
    # JWT Tokens
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "<REDACTED_JWT_TOKEN>"),
    # IPv4 Addresses (Excluding local loops 127.0.0.1, 0.0.0.0)
    (
        re.compile(r"\b(?!127\.0\.0\.1|0\.0\.0\.0)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "<REDACTED_IP_ADDR>",
    ),
    # Email Addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "<REDACTED_EMAIL>"),
]


def scrub_secrets_and_pii(text: str) -> Tuple[str, int]:
    """
    Scrub secrets and PII from a source code string.
    
    Returns:
        sanitized_text: String with sensitive tokens replaced
        redaction_count: Number of secrets redacted
    """
    sanitized = text
    total_redactions = 0

    for pattern, replacement in SECRET_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            total_redactions += len(matches)
            sanitized = pattern.sub(replacement, sanitized)

    return sanitized, total_redactions


def sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Process a dataset record dict in-place and record redaction statistics."""
    content = record.get("content", "")
    sanitized, count = scrub_secrets_and_pii(content)
    
    clean_record = dict(record)
    clean_record["content"] = sanitized
    clean_record["secrets_redacted"] = count
    return clean_record


if __name__ == "__main__":
    sample = """
    # Configure AWS and API client
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    openai_key = "sk-abcdef1234567890abcdef1234567890"
    contact_email = "developer.admin@enterprise-cloud.org"
    server_ip = "198.51.100.42"
    """

    scrubbed, count = scrub_secrets_and_pii(sample)
    print(f"Redacted {count} sensitive tokens:")
    print(scrubbed)
