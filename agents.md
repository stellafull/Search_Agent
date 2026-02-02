# Project: PAI-LangStudio Research Agent (Competition Track 2)

## Project Overview
We are building a Research Agent for the Alibaba Cloud PAI Competition (Track 2). The goal is to solve complex multi-hop reasoning questions using **LLM** and **search Tools** (Tavily, Alibaba iqs). The final deliverable is a FastAPI service deployed on PAI-EAS that takes a {question} and returns a deterministic {answer}.

### Constraints:

No Fine-tuning: Pure prompting and flow engineering.

Output Format: Strictly normalized text (lowercase, no punctuation, specific format).

Environment: Python 3.10+, LangGraph, LangChain, Pydantic, FastAPI.

Platform: Must be deployable to PAI-EAS (Standard HTTP Interface).

## Repo Structure

All code in ```/src```, the local test set store in ```/data```

TIANCHI
├──/src/
    ├── .env                  # API Keys
    ├── state.py              # Data models (Pydantic & TypedDict)
    ├── prompts.py            # Prompt templates
    ├── utils.py              # Helpers (JSON parsing, normalization) and Search tool wrapper
    ├── agents.py             # LangGraph logic (Nodes & Graph)
    └── main.py               # FastAPI entry point
├──/data/
    ├── question.jsonl        # Json style qestions
    └── agent_submit_example.jsonl        # output example
└── pyproject.toml            # uv package dependency

### python env
```bash
uv sync & source .venv/bin/activate
```
## Architecture: Iterative Plan-and-Solve
We use LangGraph to implement a DAG workflow:

Planner: Decomposes the complex question into dependent steps (e.g., Step 1 finds A, Step 2 uses A to find B).

Prep: Fills query templates with values resolved from previous steps.

Search: Executes search using Tavily API and iqs API

Extract: Reads search results and extracts specific variable candidates (JSON).

Finalize: Synthesizes the final answer based on the symbol table.

## File Plan & Ownership

This project follows a "thin entry + clear modules" rule:
- main.py only hosts HTTP IO and calls a single `solve(question)` function.
- agents.py owns the LangGraph DAG and orchestrates nodes.
- state.py defines all state schema & reducers.
- prompts.py is the single source of truth for prompts.
- utils.py contains pure helpers + tool wrappers (search, normalize, json parsing).

### /src/state.py (single source of truth for types)
- Defines:
  - Graph State schema (TypedDict / Pydantic)
  - Reducers for fan-in merges (important for parallel branches) 
- No network calls, no prompt strings.

### /src/prompts.py
- Defines:
  - Planner prompt
  - Extractor prompt (strict JSON output)
  - Finalizer prompt (normalized answer format)
- Prompts must be deterministic (temperature=0, strict formatting).

### /src/utils.py
- Defines:
  - normalize_answer()
  - safe_json_loads() / robust parsing
  - search wrappers (tavily / iqs), retry/backoff, timeouts
  - extract_model to extract subquestion answer and keep context window small (save token)
- Keep tool calls centralized so rate-limit / concurrency control is in one place.

### /src/agents.py
- Defines the LangGraph:
  - Nodes: planner → prep → search → extract → finalize
  - Edges + conditional routing
  - Graph compile() + exported `solve(question)` entry
- If parallelizing, use fan-out/fan-in patterns and reducers to merge state 

### /src/main.py
- FastAPI app + request/response models
- Calls `solve(question)` only (no business logic here)
- For higher throughput, scaling via multiple worker processes is supported

## Implementation Instructions

Generate state.py first to establish types.

Generate utils.py

Generate prompts.py

Generate agents.py

Generate main.py to expose the API

## Local Evaluation (Phase 1)

The dataset `data/question.jsonl` contains 100 questions.
For faster iteration, we run them concurrently (I/O-bound). Python's `concurrent.futures` provides thread pools for parallel task execution.

- Export `solve(question) -> answer` from agents.py
- Generate `scripts/run_local_eval.py` with `EVAL_WORKERS=8` (tune carefully to avoid API rate limits)
- Merge all results into one `data/answer.jsonl`, the example format in `data/agent_submit_example.jsonl`

Run `scripts/run_local_eval.py` with `EVAL_WORKERS=8` and save results to test our workflow