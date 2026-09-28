"""
Response Parser

What it is: turns the LLM's raw text response — formatted per the
OUTPUT FORMAT section in prompts.py, as "LABEL:\\n<content>" blocks —
into a dict of {label: content}, plus small helpers for pulling a
numbered list or a fenced code block out of one section's content.

Why required: testbench_generator.py and debug_analyzer.py both need to
turn LLM text into a validated Pydantic result (TestbenchResult /
DebugResult). Sharing this parsing logic means a malformed response is
detected the same way in both places, and a genuinely malformed response
(missing/garbled section) raises one clear error rather than each
caller's own ad hoc regex quietly doing something different.

Input: raw LLM response text (str).
Output: dict[str, str] mapping SECTION_LABEL -> its raw content.

How it connects to the next component: testbench_generator.py and
debug_analyzer.py call parse_labeled_sections() once, then pull specific
fields out of the result (via parse_numbered_list / extract_code_block)
to build their respective schema objects.

What would happen if it were removed: each generator would need its own
regex-based parsing, duplicated and possibly inconsistent between the
two modes.
"""

from __future__ import annotations

import re

from app.llm.client import LLMError

_SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Z_]*):\s*$", re.MULTILINE)
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")
_CODE_BLOCK_RE = re.compile(r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", re.DOTALL)


def parse_labeled_sections(text: str, required_labels: list[str]) -> dict[str, str]:
    """
    Split text into {label: content} using ALL_CAPS_LABEL: headers.
    Raises LLMError if any of required_labels is missing — this is what
    makes a malformed LLM response fail loudly rather than producing a
    result with silently-empty fields.
    """
    headers = list(_SECTION_HEADER_RE.finditer(text))
    if not headers:
        raise LLMError(
            "Could not parse the LLM response: no recognizable "
            "LABEL: section headers were found. Raw response:\n" + text[:500]
        )

    sections: dict[str, str] = {}
    for i, match in enumerate(headers):
        label = match.group(1)
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[label] = text[start:end].strip()

    missing = [label for label in required_labels if label not in sections]
    if missing:
        raise LLMError(
            f"LLM response is missing required section(s): {missing}. "
            f"Found sections: {list(sections.keys())}"
        )
    return sections


def parse_numbered_list(content: str) -> list[str]:
    """
    Extract items from a numbered list like "1. foo\\n2. bar". Content
    that is just "None." or "None identified." (as the prompt instructs
    the LLM to write when a list is genuinely empty) returns [].
    """
    content = content.strip()
    if not content or content.lower().startswith("none"):
        return []

    items = []
    for line in content.splitlines():
        match = _NUMBERED_ITEM_RE.match(line)
        if match:
            items.append(match.group(1).strip())
    if items:
        return items

    # Fallback: no numbered markers found but content isn't empty/"None" —
    # treat each non-blank line as one item rather than discarding real
    # content just because the model didn't number it.
    return [line.strip() for line in content.splitlines() if line.strip()]


def extract_code_block(content: str) -> str:
    """Pull code out of a ```systemverilog ... ``` fenced block. Raises
    LLMError if no fenced code block is present, since a testbench
    result with no actual code is a malformed response, not an empty
    list to fall back on."""
    match = _CODE_BLOCK_RE.search(content)
    if not match:
        raise LLMError(
            "Could not find a fenced SystemVerilog code block in the "
            "TESTBENCH_CODE section of the LLM response."
        )
    return match.group(1).strip()


def parse_comma_list(content: str) -> list[str]:
    """Extract items from a comma-separated line. 'None identified.' etc.
    return []."""
    content = content.strip()
    if not content or content.lower().startswith("none"):
        return []
    return [item.strip() for item in content.split(",") if item.strip()]
