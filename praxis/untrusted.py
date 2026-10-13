"""Untrusted-content delimiter markers shared across modules.

Delimiters marking candidate raw text as untrusted data: anything between
these markers is content to be processed, never instructions to follow.
Deliberately dependency-free so importing the markers never pulls the LLM
client chain (praxis.agents -> praxis.llm -> litellm) into light commands
like ``praxis check``.
"""

from __future__ import annotations

UNTRUSTED_START = "<<<UNTRUSTED CANDIDATE CONTENT BEGIN>>>"
UNTRUSTED_END = "<<<UNTRUSTED CANDIDATE CONTENT END>>>"

__all__ = ["UNTRUSTED_END", "UNTRUSTED_START"]
