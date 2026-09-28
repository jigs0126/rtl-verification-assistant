# RAG-Based RTL Verification Assistant

A domain-specific Retrieval-Augmented Generation (RAG) application for RTL/digital-design
verification. It has two capabilities: generating SystemVerilog testbenches and analyzing
RTL failure reports — both grounded in your actual RTL and project documentation, never
invented from the LLM's general knowledge alone.

**Version 1 is a human-in-the-loop assistant.** It does not run simulation, does not compile
RTL, does not modify RTL automatically, and does not regenerate tests after failures. You
decide whether a generated testbench or debug hypothesis is correct.

---

## 1. Project Overview

Verification engineers spend a lot of time writing boilerplate testbenches and re-deriving
context (interfaces, encodings, prior bugs) every time they investigate a failure. This
project retrieves the relevant pieces of your own RTL and documentation for a given request,
then asks an LLM to produce a grounded testbench or a structured debug analysis — with every
claim traceable back to a specific retrieved source.

## 2. Motivation

An LLM asked to "write a testbench for my ALU" with no context will invent plausible-looking
but wrong port names, operation encodings, and corner cases. RAG fixes this by retrieving your
actual `alu.sv`, `operation_table.md`, and `verification_plan.md` first, and instructing the
LLM to use only what was retrieved — with an explicit instruction to say "I don't know" rather
than invent details, and to separate facts from hypotheses when debugging.

## 3. Architecture

```
                    RTL PROJECT
                         |
              +----------+----------+
              |                     |
              v                     v
          RTL FILES             DOCUMENTS
        (rtl/, bugs/)          (docs/, examples/)
              |                     |
              +----------+----------+
                         |
                         v
                  DOCUMENT LOADER          app/ingestion/loader.py
                         |
                         v
                RTL/MARKDOWN PARSER        app/ingestion/parser.py
                (structure-aware, not
                 split at N characters)
                         |
                         v
                     CHUNKER              app/ingestion/chunker.py
                         |
                         v
                    METADATA              app/ingestion/metadata.py
                         |
                         v
                    EMBEDDINGS            app/embeddings/embedder.py
                         |
                         v
                 VECTOR DATABASE          app/retrieval/vector_store.py
                    (ChromaDB,                  (persistent,
                  local, persistent)          local to disk)
                         |
                         v
                     RETRIEVER            app/retrieval/retriever.py
                         |
             +-----------+-----------+
             |                       |
             v                       v
       TESTBENCH MODE           DEBUG MODE
             |                       |
             v                       v
     PROMPT CONSTRUCTION       PROMPT CONSTRUCTION   app/llm/prompts.py
             |                       |
             v                       v
         LLM CLIENT               LLM CLIENT         app/llm/client.py
             |                       |
             v                       v
      RESPONSE PARSER          RESPONSE PARSER       app/llm/response_parser.py
             |                       |
             v                       v
      SV TESTBENCH            DEBUG ANALYSIS
   (TestbenchResult)            (DebugResult)
             |                       |
             +-----------+-----------+
                         |
                         v
                  STREAMLIT UI          app/main.py

```

Every layer above is independent and swappable behind an interface: `EmbeddingModel`,
`LLMClient`, and the `VectorStore`/`Retriever` split. Nothing in `app/ingestion/` or
`app/retrieval/` knows it's looking at an ALU — pointing `DOCUMENTS_PATH` at a different
project (e.g. RV-CORDIX) requires no code changes, only re-indexing.

## 4. RAG Pipeline (how retrieval actually works)

```
text  -->  embedding model  -->  vector (list[float])
```

The embedding **model** produces vectors; the vector **database** only stores and searches
them by similarity — it doesn't itself understand anything. Retrieval works by embedding the
incoming query the same way the corpus was embedded, then finding the nearest vectors by
cosine distance.

**Keyword search vs. semantic retrieval:** keyword search matches literal substrings — a
query for "subtraction" would not match a chunk containing only "SUB operation" or "a - b",
because none of those share the literal token "subtraction". Semantic retrieval compares
*meaning* via embedding vectors, so a well-trained model places "How is subtraction
implemented?" close to SUB/`a - b`/two's-complement chunks even without shared vocabulary.
This is what makes RAG retrieval meaningfully better than grep here — engineers ask questions
in natural language, not RTL identifier syntax.

**Caveat (see Limitations):** that semantic benefit is only real with a trained embedding
model (`all-MiniLM-L6-v2`). This sandbox's demonstration run used a deterministic
hashing fallback instead, which is lexical/character-overlap based — closer to keyword search
than true semantic retrieval. The retrieval *mechanism* is correct and model-agnostic; its
retrieval *quality* depends entirely on which embedding backend is active.

## 5. Testbench Generation Workflow

1. You provide `{module, requirement}` (e.g. "Generate a testbench for the ALU covering all
   operations and important boundary cases").
2. `Retriever.query()` embeds the requirement and retrieves the top-`k` most relevant chunks.
3. `prompts.build_testbench_prompt()` assembles a four-section prompt: SYSTEM INSTRUCTIONS
   (hallucination-control rules) / PROJECT CONTEXT (retrieved chunks, each labeled with its
   source) / USER REQUEST / OUTPUT FORMAT.
4. `LLMClient.complete()` sends it.
5. `response_parser.py` splits the labeled-section response into
   `{test_scenarios, assumptions, warnings, testbench_code}`.
6. `TestbenchResult` is returned and rendered — always carrying
   **"Generated artifact — not simulation-verified."** regardless of what the model itself
   said, and always listing which source files it was grounded in.

## 5b. RTL Workflow (spec -> RTL -> testbench -> debug)

The **RTL Workflow** tab is the end-to-end loop for a design you bring yourself
(it does not depend on the indexed knowledge base):

1. **Specification -> RTL.** Paste a spec or example; the app returns SystemVerilog
   RTL plus the interface table, design decisions and assumptions.
2. **Verification plan -> testbench.** Paste your plan; the app writes a
   self-checking testbench for the RTL. Every check prints `[PASS] name` or
   `[FAIL] name: expected=... observed=...` and a final `SUMMARY:` line, so the
   log can be parsed reliably. A plan-coverage list shows which plan items were
   (or were not) implemented.
3. **Simulate in your own tool, paste the console log back** (pass or fail).
   The app first extracts facts *without the LLM* (pass/fail counts, failing
   lines, compile/runtime error lines), then asks the LLM for root causes, each
   tagged `[RTL]` / `[TESTBENCH]` / `[SPEC]` / `[UNKNOWN]`, prioritized fixes, and
   (when small enough) a full corrected file. Apply it, re-simulate, paste the new log.

The RTL and testbench boxes are editable, so the loop can be repeated. The app
never compiles or simulates anything itself: generated code is unverified, and
root causes are hypotheses until your simulator confirms the fix.
Code: `app/verification/rtl_workflow.py`, `app/llm/workflow_prompts.py`,
`app/schemas/workflow.py`; tests: `tests/test_rtl_workflow.py`.

## 6. Debugging Workflow

1. You provide `{module, failure_report}` (test name, operation, inputs, expected/observed
   values, notes).
2. Retrieval uses the failure report itself as the query — its operation name and symptom
   description surface relevant RTL, spec, and **historical bug reports** (`bugs/*.md`).
3. The prompt instructs the LLM to label every claim **FACT** (directly supported by the
   report or retrieved context), **HYPOTHESIS** (a plausible but unconfirmed explanation), or
   **UNKNOWN**. It's explicitly told not to conclude a match to a historical bug just because
   one looks similar, and never to claim simulation/waveform access it doesn't have.
4. `DebugResult` keeps `evidence` (facts) and `possible_root_causes` (hypotheses) as separate
   fields, so the UI can never accidentally present a hypothesis as a fact by rendering them
   identically.

## 7. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | — |
| Vector store | ChromaDB (local, persistent) | Runs embedded, no server; stores metadata alongside vectors natively; simple API that doesn't hide what's happening |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2`, behind an `EmbeddingModel` interface | Local, no per-query API cost after the model weights are downloaded once |
| LLM | Google Gemini API (free tier via AI Studio), behind the `LLMClient` interface | Application code never imports the SDK directly; another provider can be added as one more `LLMClient` class |
| UI | Streamlit | Fast to build an inspectable, interactive UI |
| Config | `pydantic-settings` + `.env` | Typed, validated, single source of truth |
| Schemas | Pydantic | A malformed LLM response fails loudly (validation error) instead of silently corrupting output |

## 8. Installation

```bash
git clone <this repo>
cd rtl-verification-assistant
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_API_KEY to a key from aistudio.google.com
```

## 9. Environment Variables (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | *(empty)* | Gemini API key. Required for real generation. |
| `LLM_MODEL` | `gemini-3.8-flash` | Gemini model name. Check Google AI Studio for current names. |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model name. |
| `VECTOR_DB_PATH` | `data/vector_store` | Where ChromaDB persists to disk. |
| `DOCUMENTS_PATH` | `.` | Project root to index (`rtl/`, `docs/`, `bugs/`, `examples/` under it). |
| `LOG_PATH` | `data/logs` | Where query logs are written. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking parameters. |
| `TOP_K` | `4` | Chunks retrieved per query. |

`LLM_API_KEY` is never printed or logged in full — `Settings.masked_api_key()` shows only the
first 5 and last 4 characters, and `app/utils/logging.py`'s query log has no field capable of
holding it.

## 10. How to Index Documents

Indexing is an **explicit, visible action** — it never happens silently on app start (§22).

```bash
python3 -c "
from pathlib import Path
from app.config.settings import settings
from app.embeddings.embedder import get_embedding_model
from app.retrieval.vector_store import VectorStore
from app.retrieval.indexer import index_project

model = get_embedding_model(settings.embedding_model)
store = VectorStore(path=Path(settings.vector_db_path))
count = index_project(Path(settings.documents_path), store, model, project_name='alu')
print(f'Indexed {count} chunks.')
"
```

Or use the **"Re-index Knowledge Base"** button in the Streamlit sidebar. Re-indexing is
idempotent: chunks are upserted by a deterministic `chunk_id`, so running it repeatedly never
creates duplicates — it only updates chunks whose source text actually changed.

## 11. How to Run

```bash
python run.py
# equivalent to: streamlit run app/main.py
```

Three tabs: **Testbench Generator**, **Debug Assistant**, **Knowledge Base** (shows indexed
file counts, lets you run a raw retrieval query and inspect results, shows the recent query
log).

## 12. How to Run Tests

```bash
python3 -m pytest -v
```

**154 tests, all passing.** Coverage spans every layer: loader, parser (RTL structure
preservation), chunker, metadata, embedder (interface contract + model-mixing guard),
vector store (persistence, idempotent upsert, model-mismatch rejection), retriever, prompt
construction, response parsing, testbench generator, debug analyzer, query logging, and a
dedicated `tests/test_integration_e2e.py` that runs the **real** pipeline (real ingestion,
real chunking, real embedding, real ChromaDB, real retrieval, real prompt construction) end
to end against the actual ALU corpus — only the LLM call itself is faked, via `FakeLLMClient`.

## 13. Example Queries

**Testbench Generator:**
> Module: `alu`
> Requirement: "Generate a SystemVerilog testbench for the ALU covering all operations and
> important boundary cases."

**Debug Assistant:**
> Module: `alu`
> Failure report:
> ```
> Test: alu_sub_01
> Operation: SUB
> A = 20
> B = 10
> Expected = 10
> Observed = 4294967286 (0xFFFFFFF6)
> ```

## 14. Limitations

- **No simulation.** Generated testbenches are never compiled or run. "Not
  simulation-verified" is always shown, regardless of what the LLM says.
- **No embedding model download in the original build sandbox.** `all-MiniLM-L6-v2`
  requires network access to `huggingface.co`, which this project's sandboxed build/test
  environment couldn't reach (only pypi/github/npm/crates were allowlisted). `get_embedding_model()`
  auto-falls-back to `DeterministicHashingEmbeddingModel` — a deterministic, dependency-free,
  **non-semantic** feature-hashing embedding — whenever `sentence-transformers` can't actually
  produce vectors, with a loud `UserWarning`. **On your own machine, with normal network
  access, `pip install -r requirements.txt` gives you the real model and nothing else changes**
  — same interface, same pipeline code.
- **No LLM API key in the original build sandbox**, so live generation could not be
  demonstrated there either. `FakeLLMClient` (clearly labeled, never silently substituted)
  stood in for demonstration purposes. On your machine, set a real `LLM_API_KEY` and
  `GeminiLLMClient` is used automatically.
- **Retrieval quality is only as good as the active embedding backend.** With the hashing
  fallback, retrieval is closer to keyword/lexical matching than true semantic search.
- **No autonomous behavior.** No feedback loop, no auto-regeneration, no multi-agent
  orchestration, no fine-tuning. Every output is a single-shot, human-reviewed artifact.
- **No formal correctness guarantee on parsing.** The RTL parser is a lightweight heuristic
  (regex + line-state), not a full SystemVerilog grammar — sufficient for a clean ALU-scale
  module, but not a substitute for a real parser on arbitrarily complex RTL.

## 15. Extending to RV-CORDIX

Nothing in `app/ingestion/`, `app/embeddings/`, or `app/retrieval/` references "ALU" — the
project name is a parameter (`project_name="alu"` in `index_project()`), and the corpus root
is a setting (`DOCUMENTS_PATH`). To point the system at RV-CORDIX:

1. Set `DOCUMENTS_PATH` to the RV-CORDIX project root (containing its own `rtl/`, `docs/`,
   `bugs/`, `examples/` subdirectories — ALU, register file, immediate generator, control
   unit, Booth multiplier, divider, CORDIC, PC unit, instruction memory, pipeline RTL, RISC-V
   docs, verification docs, test cases, bug reports).
2. Use a different `collection_name` (or a different `VECTOR_DB_PATH`) so RV-CORDIX chunks
   don't mix with the ALU demo collection — `VectorStore` already enforces that two different
   embedding-model runs can't be mixed in one collection; a fresh collection per project
   keeps corpora cleanly separated too.
3. Re-run indexing. No other code changes are required — the RTL parser already handles
   arbitrary modules/always-blocks/localparams generically, not just the ALU's five operations.

## 16. Interview Q&A Crib Notes

- **What is RAG?** Retrieving external knowledge (your RTL/docs) at inference time and
  inserting it into the prompt, rather than relying on the LLM's pretrained knowledge alone.
- **What is an embedding?** A fixed-length numeric vector representing a piece of text's
  meaning, produced by a trained model, such that texts with similar meaning produce vectors
  that are close together (by cosine similarity).
- **Why a vector database?** To store many embeddings and search them by similarity
  efficiently, rather than comparing a query against every chunk's vector one at a time in
  application code.
- **How does semantic retrieval work?** Embed the query the same way the corpus was embedded,
  then return the nearest vectors by cosine distance — see §4 above.
- **What exactly gets sent to the LLM?** A four-section prompt: system instructions
  (hallucination-control rules), the retrieved chunks (each labeled with its source file and
  section), the user's request, and the required output format — see `app/llm/prompts.py`.
- **How does testbench generation work?** §5 above.
- **How does debugging work?** §6 above.
- **What prevents hallucinated RTL information?** The prompt explicitly instructs the LLM to
  use only the supplied PROJECT CONTEXT for project-specific facts and to say so explicitly
  when context is insufficient — see the `_SHARED_RULES` constant in `app/llm/prompts.py`.
- **Why didn't you fine-tune the model?** Fine-tuning changes model *parameters* via training
  data; RAG retrieves external knowledge at *inference time* with no training step — far
  cheaper, and the knowledge base updates instantly when RTL/docs change, with no retraining.
- **Why isn't this an autonomous agent yet?** Version 1 uses no tool calling, no simulation,
  and no feedback loop — every artifact is single-shot and human-reviewed. See §14 above and
  the phase roadmap below.

## 17. Future Work (not implemented — by design, see project scope)

| Phase | Extension |
|---|---|
| 2 | Verilator/Vivado/other simulator integration |
| 3 | Automatically compile generated testbenches |
| 4 | Run simulation and collect results |
| 5 | Coverage analysis |
| 6 | generate → simulate → analyze → regenerate feedback loop |
| 7 | Tool calling |
| 8 | Convert into an agentic verification system |
| 9 | RISC-V assembly test generation |
| 10 | Replace the ALU knowledge base with RV-CORDIX (see §15 — the plumbing is already in place) |
