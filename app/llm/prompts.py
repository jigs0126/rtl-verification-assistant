"""
Prompt Construction

What it is: the single place that assembles the four-section prompt
structure (SYSTEM INSTRUCTIONS / PROJECT CONTEXT / USER REQUEST / OUTPUT
FORMAT) sent to the LLM, for both testbench generation and debug
analysis.

Why required: hallucination-control rules ("don't invent ports", "state
uncertainty", "separate fact from hypothesis") need to live in exactly
one place each, not be re-typed slightly differently in
testbench_generator.py and debug_analyzer.py. Centralizing prompt text
also means the "what exactly gets sent to the LLM?" interview question
has one obvious answer.

Input: a RetrievedContext (from retriever.py) plus task-specific request
fields (module + requirement, or module + failure_report).
Output: (system_prompt: str, user_prompt: str) — kept as two separate
strings because LLM APIs commonly take `system` as its own field rather
than folding it into the message content.

How it connects to the next component: LLMClient.complete(user_prompt,
system=system_prompt) sends exactly what this module built. The response
text is then parsed by testbench_generator.py / debug_analyzer.py.

What would happen if it were removed: each caller would build its own
ad hoc prompt string, making the hallucination-control rules easy to
accidentally omit or contradict between the two modes.
"""

from __future__ import annotations

from app.schemas.models import RetrievedContext

# Shared across both modes — the rules from §24 (hallucination control).
_SHARED_RULES = """\
RULES:
- Use only the supplied PROJECT CONTEXT for project-specific facts (signal
  names, port lists, operation encodings, architecture details). Do not
  invent RTL interfaces, signal names, operation encodings, or
  project-specific architecture not present in the retrieved context.
- If the retrieved context does not contain enough information to answer
  confidently, say so explicitly rather than guessing.
- Never claim that generated code has been simulated, compiled, or
  verified. It has not been. Never claim a bug is confirmed unless the
  supplied evidence actually proves it.
- Distinguish clearly between facts (directly supported by the supplied
  evidence) and hypotheses (plausible but unconfirmed inferences)."""


def _format_context(context: RetrievedContext) -> str:
    """Render retrieved chunks as labeled [Source: ...] blocks."""
    if context.is_empty:
        return (
            "(No relevant project context was retrieved for this query. "
            "Do not invent project-specific details to fill this gap — "
            "state explicitly that no grounding context was found.)"
        )
    blocks = []
    for result in context.results:
        chunk = result.chunk
        header = f"[Source: {chunk.source} | Section: {chunk.section} | Distance: {result.score:.4f}]"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_testbench_prompt(module: str, requirement: str, context: RetrievedContext) -> tuple[str, str]:
    """Assemble the (system, user) prompt pair for testbench generation."""
    system_prompt = f"""You are an RTL verification assistant helping generate a \
SystemVerilog testbench for the "{module}" module.

{_SHARED_RULES}
- Use the actual retrieved module interface — do not invent ports.
- Use SystemVerilog `logic` types appropriately and instantiate the DUT
  correctly based on the retrieved interface.
- Cover the documented operations and include meaningful corner cases
  drawn from the retrieved verification plan / specification, not
  invented ones, when a verification plan is present in the context.
- Include readable signal naming and comments.
- Clearly state any assumptions you had to make.

OUTPUT FORMAT:
Respond with exactly these four sections, in this order, each on its own
line starting with the label shown:

TEST_SCENARIOS:
<a numbered list of the test scenarios you are covering>

ASSUMPTIONS:
<a numbered list of assumptions you made, or "None." if none>

WARNINGS:
<a numbered list of caveats, or "None." if none — always include at \
least the fact that this testbench has not been simulation-verified>

TESTBENCH_CODE:
```systemverilog
<the complete SystemVerilog testbench>
```"""

    user_prompt = f"""PROJECT CONTEXT:

{_format_context(context)}

USER REQUEST:

Generate a SystemVerilog testbench for module "{module}" that satisfies \
this requirement:

{requirement}"""

    return system_prompt, user_prompt


def build_debug_prompt(module: str, failure_report: str, context: RetrievedContext) -> tuple[str, str]:
    """Assemble the (system, user) prompt pair for debug analysis."""
    system_prompt = f"""You are an RTL verification assistant helping debug a reported \
failure in the "{module}" module.

{_SHARED_RULES}
- For every claim, label it explicitly as one of: FACT (directly
  supported by the supplied failure report or retrieved context),
  HYPOTHESIS (a plausible but unconfirmed explanation), or UNKNOWN
  (information that is simply not available from what was supplied).
- Do not conclude the failure matches a previous bug just because a
  retrieved historical bug report looks similar — you may note the
  resemblance, but must not treat it as confirmation.
- You have not run simulation and cannot inspect waveforms — do not
  imply otherwise.

OUTPUT FORMAT:
Respond with exactly these sections, in this order, each on its own line
starting with the label shown:

FAILURE_SUMMARY:
<one or two sentences>

EXPECTED_BEHAVIOR:
<what should have happened, per the spec/context>

OBSERVED_BEHAVIOR:
<what the failure report says actually happened>

INTERPRETATION:
<your reading of the discrepancy>

RELEVANT_MODULES:
<comma-separated module names>

RELEVANT_SIGNALS:
<comma-separated signal names, or "None identified." if none>

POSSIBLE_ROOT_CAUSES:
<a numbered list, each item labeled [HYPOTHESIS] since none are confirmed \
unless the evidence proves them>

EVIDENCE:
<a numbered list of the specific facts, labeled [FACT], that support your \
interpretation>

RECOMMENDED_INVESTIGATION:
<a numbered list of concrete next manual-investigation steps>

CONFIDENCE:
<low / medium / high, with one sentence on why>"""

    user_prompt = f"""PROJECT CONTEXT:

{_format_context(context)}

USER REQUEST:

Analyze this failure report for module "{module}":

{failure_report}"""

    return system_prompt, user_prompt
