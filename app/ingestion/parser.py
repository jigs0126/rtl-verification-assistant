"""
Parser

What it is: turns a RawDocument's raw text into a list of ParsedSection
objects — structurally meaningful pieces, before size-based chunking.

Why required: splitting raw text at arbitrary character boundaries can
cut a module declaration or an always block in half, which destroys the
RTL's meaning as a retrievable unit. Markdown documents similarly read
much better as retrieval units when split at heading boundaries rather
than mid-sentence.

Input: RawDocument (from loader.py)
Output: list[ParsedSection]

How it connects to the next component: chunker.py applies size-based
splitting *within* each section (never across section boundaries), so
a section that's already small enough becomes exactly one chunk.

What would happen if it were removed: chunker.py would have to operate
on raw, unstructured text and would regularly cut RTL constructs and
markdown headings mid-block, producing chunks that don't make sense in
isolation.
"""

from __future__ import annotations

import re

from app.schemas.models import ParsedSection, RawDocument


def parse_document(doc: RawDocument) -> list[ParsedSection]:
    """Dispatch to the appropriate structure-aware parser by file_type."""
    if doc.file_type in ("systemverilog", "verilog"):
        return _parse_rtl(doc)
    if doc.file_type == "markdown":
        return _parse_markdown(doc)
    return _parse_text(doc)


def _section(doc: RawDocument, section: str, text: str, module: str | None) -> ParsedSection:
    return ParsedSection(
        source=doc.source,
        file_type=doc.file_type,
        project=doc.project,
        section=section,
        text=text,
        module=module,
    )


# ============================================================== RTL =====

_MODULE_RE = re.compile(r"\bmodule\s+(\w+)")
_ALWAYS_START_RE = re.compile(r"^\s*always(_comb|_ff|_latch)?\b")
_ASSIGN_RE = re.compile(r"^\s*assign\b")
_PARAM_RE = re.compile(r"^\s*(localparam|parameter)\b")
_ENDMODULE_RE = re.compile(r"^\s*endmodule\b")
_CASE_TARGET_RE = re.compile(r"case\s*\(\s*(\w+)")


def _parse_rtl(doc: RawDocument) -> list[ParsedSection]:
    """
    Walk the RTL file line by line, grouping lines into named structural
    sections: header comments, the module/port declaration, parameter
    declarations, each always block (labeled by its case-statement
    target signal when detectable), continuous assignments, and the
    endmodule line. This is a lightweight heuristic parser (regex +
    line-state), not a full SystemVerilog grammar — sufficient for
    keeping RTL constructs intact without the complexity of a real
    parser, per the project's Phase 3 scope.
    """
    lines = doc.content.splitlines()
    module_match = _MODULE_RE.search(doc.content)
    module_name = module_match.group(1) if module_match else None

    sections: list[ParsedSection] = []
    i, n = 0, len(lines)

    # 1. Header comments before the module keyword.
    header_lines: list[str] = []
    while i < n and "module" not in lines[i]:
        header_lines.append(lines[i])
        i += 1
    if any(l.strip() for l in header_lines):
        sections.append(_section(doc, "header_comments", "\n".join(header_lines), module_name))

    # 2. Module declaration + port list, through the closing ");".
    if i < n:
        decl_lines: list[str] = []
        while i < n:
            decl_lines.append(lines[i])
            if ");" in lines[i]:
                i += 1
                break
            i += 1
        sections.append(_section(doc, "module_declaration", "\n".join(decl_lines), module_name))

    # 3. Remaining body: group into parameter block / always blocks /
    #    continuous assignments / other, each as its own section.
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines and any(l.strip() for l in current_lines):
            sections.append(_section(doc, current_label or "body", "\n".join(current_lines), module_name))

    while i < n:
        line = lines[i]

        if _ENDMODULE_RE.match(line):
            flush()
            sections.append(_section(doc, "endmodule", line, module_name))
            current_label, current_lines = None, []
            i += 1
            continue

        if _PARAM_RE.match(line):
            if current_label != "parameter_declarations":
                flush()
                current_label, current_lines = "parameter_declarations", []
            current_lines.append(line)
            i += 1
            continue

        if _ALWAYS_START_RE.match(line):
            flush()
            block_lines = [line]
            i += 1
            while (
                i < n
                and not _ALWAYS_START_RE.match(lines[i])
                and not _ASSIGN_RE.match(lines[i])
                and not _ENDMODULE_RE.match(lines[i])
            ):
                block_lines.append(lines[i])
                i += 1
            block_text = "\n".join(block_lines)
            case_match = _CASE_TARGET_RE.search(block_text)
            label = f"always_block[{case_match.group(1)}]" if case_match else "always_block"
            sections.append(_section(doc, label, block_text, module_name))
            current_label, current_lines = None, []
            continue

        if _ASSIGN_RE.match(line):
            if current_label != "continuous_assignments":
                flush()
                current_label, current_lines = "continuous_assignments", []
            current_lines.append(line)
            i += 1
            continue

        if current_label is None:
            current_label = "body"
        current_lines.append(line)
        i += 1

    flush()
    return sections


# =========================================================== markdown ===

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _parse_markdown(doc: RawDocument) -> list[ParsedSection]:
    """Split on heading lines; each heading starts a new section that
    includes the heading text and everything until the next heading."""
    lines = doc.content.splitlines()
    sections: list[ParsedSection] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines and any(l.strip() for l in current_lines):
            sections.append(_section(doc, current_heading, "\n".join(current_lines), None))

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    if not sections:
        sections.append(_section(doc, "Document", doc.content, None))
    return sections


# =============================================================== text ===


def _parse_text(doc: RawDocument) -> list[ParsedSection]:
    """Plain text files are not sub-parsed — the whole file is one section."""
    return [_section(doc, "Document", doc.content, None)]
