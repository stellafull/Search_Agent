# Project: PAI-LangStudio Research Agent (Competition Track 2)

## Project Overview

We are building a **Research Agent** for the Alibaba Cloud PAI Competition (Track 2). The goal is to solve **complex multi-hop reasoning questions** using an **LLM** and **search tools** (**Tavily** + **Alibaba IQS**). The final deliverable is a **FastAPI** service deployed on **PAI-EAS** that takes `{question}` and returns a deterministic `{answer}`.

### Constraints

- **No Fine-tuning**: Pure prompting and flow engineering.
- **Output Format**: Strictly normalized text (lowercase, no punctuation, specific format).
- **Environment**: Python 3.10+, LangGraph, LangChain, Pydantic, FastAPI.
- **Platform**: Must be deployable to PAI-EAS (Standard HTTP Interface).
- **Determinism**: Temperature=0, bounded loops/budgets, typed node contracts.

### Example (multi-hop question)

```json
{
  "question": "一位物理学领域的学者为一种经典棋盘游戏设计的评分系统，后来被一家北美游戏公司广泛应用于其一款多人在线战术竞技游戏中。这家公司的母公司是一家亚洲科技巨头，该巨头在21世纪10年代完成了对前者的全资收购，并涉足量子计算等前沿科技领域。在这家北美公司开发的另一款第一人称射击游戏中，有一件适合近距离作战的武器，其名称与上述亚洲巨头代理发行的一款格斗手游中的一名在登场角色中年龄偏大的武术教官角色相同。这款格斗手游的名字是什么？",
  "answer": "魂武者"
}
```

---

## Repo Structure

All code in `/src`, local test set stored in `/data`.

```
TIANCHI
├── /src/
│   ├── .env                  # API Keys
│   ├── state.py              # Graph state schema + shared domain models
│   ├── prompts.py            # Prompt text + structured output schemas (TypedDict/BaseModel)
│   ├── utils.py              # Helpers + Search wrappers + LLM factory + assertions + normalization
│   ├── agents.py             # LangGraph logic (Nodes & Graph) + solve(question)
│   └── main.py               # FastAPI entry point (Phase 2)
├── /data/
│   ├── question.jsonl        # Json style questions
│   └── agent_submit_example.jsonl
└── pyproject.toml            # uv package dependency
```

### Python env

```bash
uv sync && source .venv/bin/activate
```

---

## Architecture: Self-Ask + ReAct + Reflexion (Schema-free / Need-driven)

### Why NOT hard-code a slot schema?

The dataset is diverse. Hard-coding slots (e.g., “parent company / weapon / mobile game”) does not generalize and increases error propagation.
Instead, we use a **schema-free, need-driven loop**:

- **Self-Ask**: identify the next missing “Need” (a follow-up question).
- **ReAct (within-hop)**: search → observe → refine queries/tools.
- **Baleen-style condensation**: keep a compact, verified note set to control noise.
- **Reflexion**: when stuck, write actionable strategy feedback for the next few hops.

### Flow (per-hop loop)

**1) Gap Analyzer (Planner / Need Inducer)**

- **Input**: `question + known_facts/notes + previous_reflection (optional)`
- **Output**: one `current_gap` (single most critical missing need), or `READY_TO_ANSWER`
- **Rule**: if a reflection exists (last attempt failed), planner must change strategy (e.g., switch entity, change language, add constraints).

**2) Dual Search (Executor / ReAct Retrieval)**

- **Input**: `current_query` (derived from `current_gap`)
- **Logic**: execute **Tavily** and/or **IQS** (optionally parallel), then merge, deduplicate, truncate.
- **Within-hop ReAct**: if results are weak, rewrite query and/or switch tool up to `MAX_TOOL_CALLS_PER_HOP`.

**3) Fact Extractor & Internal Critic (Verifier)**

- **Input**: `current_gap + search_results`
- **Action**: extract candidate facts + evidence, then judge:
  1) **Presence**: is the fact explicitly supported by the text? (anti-hallucination)
  2) **Relevance**: does it directly answer the gap?
- **Outcome**:
  - **PASS / CONFIRMED**: add to `known_facts`, clear reflection, mark progress
  - **WEAK / FAIL**: produce actionable feedback (`missing`, `retry_hint`) and optionally reflection

**4) Loop Termination**

- **Success**: planner returns `READY_TO_ANSWER`
- **Max hops**: if `loop_count > MAX_HOPS`, return best-effort answer (with deterministic normalization)

---

## Strongly Typed Node Outputs (No “JSON-only prompts”)

### Core policy

Prompts should **not** rely on “You must output JSON with keys ...” as the main enforcement mechanism.
Instead, **every LLM node output is constrained by schema** via LangChain’s `with_structured_output` (TypedDict or Pydantic BaseModel).

This gives:

- deterministic parsing (no fragile JSON string parsing)
- validation errors become structured feedback signals
- simpler routing based on typed fields

---

## File Plan & Ownership (Clear Logic Boundaries)

### Guiding principles

1) **Thin entry**: `main.py` only handles HTTP I/O and calls `solve(question)`.
2) **Single ownership**: each module owns one responsibility and exposes stable interfaces.
3) **Typed contracts**: nodes communicate via `AgentState` + structured outputs (no raw JSON strings).

### /src/state.py — Graph State & Shared Domain Models

**Owns**

- `AgentState` (TypedDict): the canonical LangGraph state keys
  - e.g., `question`, `known_facts`, `current_gap`, `queries`, `search_results`, `reflection`, `loop_count`, `final_answer`
- Cross-node shared models (Pydantic recommended):
  - `SearchResultItem`, `Evidence`, `FailureEvent`, `ToolStep`, etc.
- Reducers / merge helpers (if we fan-out/fan-in)

**Does NOT own**

- prompt strings
- tool calling logic
- node-specific output schemas (those live in `prompts.py`)

### /src/prompts.py — Prompt Text + Node I/O Schemas (Contracts)

**Owns**

- Prompt text for each node (role + reasoning policy; no JSON-format instructions)
- **Node output schemas** (TypedDict or BaseModel) used by `with_structured_output`, e.g.:
  - `GapAnalysisResult`
  - `QueryPlanResult`
  - `CritiqueResult`
  - `ReflectionResult`
  - `FinalAnswerResult`

**Rationale**

- `prompts.py` is the **contract layer**: it defines what each node must produce.
- `agents.py` binds `llm.with_structured_output(<schema>)` using these schemas.

### /src/utils.py — Tools, Robustness, and Guardrails

**Owns**

- Search wrappers (centralized):
  - `search_tavily(query, ...)`
  - `search_iqs(query, page=1, ...)`
- Merge/dedup/truncate results (keep context small)
- Rate-limit friendliness: retry/backoff, timeouts
- `normalize_answer()` (competition strict output)
- **Assertions/backtracking checkers** (DSPy-style):
  - query too long
  - query too similar to previous queries
  - missing required keywords
- Optional fallback parsing helpers (only if a model/tool returns unstructured text)

**Does NOT own**

- graph routing / orchestration
- prompt templates / schemas

### /src/agents.py — LangGraph Orchestration + Node Implementations

**Owns**

- Model binding (typed):
  - `planner_llm = llm.with_structured_output(GapAnalysisResult)`
  - `critic_llm  = llm.with_structured_output(CritiqueResult)`
- Nodes (pure orchestration + state updates):
  - `analyze_gap`: runs planner_llm, updates `current_gap` / termination
  - `build_query`: generates 1–3 queries and tool hints
  - `search`: calls utils wrappers, stores results
  - `extract_critique`: runs critic_llm, updates `known_facts` or `reflection`
  - `reflect`: writes reflection strategy when stuck
  - `finalize`: produces final answer object
- Conditional routing (simple boolean checks on typed fields)
- Graph compile() + exported `solve(question)`

### /src/main.py — FastAPI Service (Phase 2)

**Owns**

- HTTP models and endpoints
- Calls `solve(question)` only
- Health endpoint for EAS

---

## Reflexion Without Gold Labels (No Standard Answers Provided)

The dataset may not include reference answers. Reflexion can still run using **proxy feedback signals** derived from the environment/tool observations and internal validators.

### Proxy feedback signals we use

- **Evidence gap**: search results do not contain the needed entity/attribute
- **Contradiction**: extracted facts conflict across sources
- **No-delta streak**: multiple hops without adding confirmed facts
- **Query drift/repetition**: assertions fail (too similar / too broad / missing must-include)
- **Schema validation errors**: structured output fails validation

### How Reflexion is used

- Inputs: last `current_gap`, recent queries, tool traces, and proxy feedback signals
- Output: short actionable instructions (TTL: applies to next 2–3 hops only), e.g.:
  - “switch to English alias + add constraint keyword”
  - “use IQS rerank + page=2”
  - “disambiguate entity name before continuing”

---

## Implementation Instructions (Recommended Order)

1) `state.py` — establish state + shared models
2) `prompts.py` — establish prompt text + node schemas (TypedDict/BaseModel)
3) `utils.py` — search wrappers + assertions + normalization
4) `agents.py` — bind LLM with structured output, build LangGraph, export `solve()`
5) `main.py` — FastAPI wrapper (Phase 2)

---

## Local Evaluation (Phase 1)

Dataset: `data/question.jsonl` (100 questions)

### Plan

- Export `solve(question) -> answer` from `agents.py`
- Create `scripts/run_local_eval.py` 
  - Use `ThreadPoolExecutor` (I/O bound)
  - `BATCH_SIZE=10` (tune to avoid API rate limits)
  - Save results to `data/answer.jsonl` following `agent_submit_example.jsonl` format
- Create `scripts/run_one_eval.py` 
  - test first question in `data/question.jsonl` to test workflow
### Run

- Run `scripts/run_one_eval.py` to test the workflow end-to-end.

### Debug checklist

- Looping: check `loop_count`, no-delta streak, and reflection triggers
- Hallucination: only accept facts with explicit evidence
- Drifting queries: ensure assertions/backtracking works
- Output format: always apply `normalize_answer()` before returning

