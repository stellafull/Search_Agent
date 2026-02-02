from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

from src.prompts import (
    FINAL_PROMPT,
    PLANNER_FEWSHOT,
    PLANNER_PROMPT,
    QUERYGEN_PROMPT,
    REASONER_PROMPT,
)
from src.state import (
    GraphState,
    PlannerOutput,
    QueryGenOutput,
    ReasonerOutput,
    SearchPlanItem,
    SubQuestion,
)
from src.utils import compact_search_results, extract_model, normalize_answer, search_all

_ = load_dotenv(find_dotenv())

MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen3-VL-235B-A22B-Thinking")
MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MAX_ITERS = int(os.getenv("MAX_ITERS", "4"))
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
SEARCH_MAX_TOTAL = int(os.getenv("SEARCH_MAX_TOTAL", "30"))
EXTRACT_MAX_ITEMS = int(os.getenv("EXTRACT_MAX_ITEMS", "12"))
EXTRACT_SNIPPET_CHARS = int(os.getenv("EXTRACT_SNIPPET_CHARS", "200"))
EXTRACT_MAX_TOKENS = int(os.getenv("EXTRACT_MAX_TOKENS", "800"))


def _init_llm():
    return init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0,
    )


planner_llm = _init_llm()
querygen_llm = _init_llm()
extract_llm = extract_model.bind(max_tokens=EXTRACT_MAX_TOKENS)
final_llm = _init_llm()

planner_structured = planner_llm.with_structured_output(PlannerOutput)
querygen_structured = querygen_llm.with_structured_output(QueryGenOutput)
extract_structured = extract_llm.with_structured_output(ReasonerOutput)


def _dump_model(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _parse_subquestions(raw: Dict[str, Any]) -> List[SubQuestion]:
    subqs: List[SubQuestion] = []
    seen: set[str] = set()
    for item in raw.get("subquestions", []) or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        subqs.append(
            {
                "id": str(item.get("id", f"SQ{len(subqs)+1}")),
                "question": question,
            }
        )
    return subqs


async def planner_node(state: GraphState) -> Dict[str, Any]:
    prompt = PLANNER_PROMPT.invoke(
        {
            "question": state.get("question", ""),
            "context": state.get("context", ""),
            "fewshot": PLANNER_FEWSHOT,
        }
    )
    response = await planner_structured.ainvoke(prompt)
    parsed = response if isinstance(response, PlannerOutput) else PlannerOutput.model_validate(response)
    status = str(parsed.status).strip().lower()
    raw_subquestions = parsed.subquestions
    subquestions = [] if status == "final" else _parse_subquestions({"subquestions": [_dump_model(sq) for sq in raw_subquestions]})
    final_answer = parsed.final_answer.strip() if status == "final" else ""
    return {
        "planner": {"status": status, "final_answer": final_answer},
        "subquestions": subquestions,
        "draft_final_answer": final_answer,
    }


async def prep_node(state: GraphState) -> Dict[str, Any]:
    subquestions = state.get("subquestions", []) or []
    plans: List[SearchPlanItem] = []

    async def _gen(subq: SubQuestion) -> SearchPlanItem:
        prompt = QUERYGEN_PROMPT.invoke(
            {"subquestion": subq.get("question", ""), "context": state.get("context", "")}
        )
        resp = await querygen_structured.ainvoke(prompt)
        data = resp if isinstance(resp, QueryGenOutput) else QueryGenOutput.model_validate(resp)
        return {
            "subquestion_id": subq.get("id", ""),
            "iqs_queries": [q for q in data.iqs_queries if isinstance(q, str)],
            "tavily_queries": [q for q in data.tavily_queries if isinstance(q, str)],
            "notes": str(data.notes),
        }

    if subquestions:
        plans = await asyncio.gather(*[_gen(sq) for sq in subquestions])

    return {"search_plan": plans, "subquestions": [], "planner": {}}


async def search_node(state: GraphState) -> Dict[str, Any]:
    plans = state.get("search_plan", []) or []
    all_results: List[Dict[str, Any]] = []

    for plan in plans:
        iqs_queries = plan.get("iqs_queries", []) or []
        tavily_queries = plan.get("tavily_queries", []) or []
        results = await search_all(
            iqs_queries=iqs_queries,
            tavily_queries=tavily_queries,
            max_results=SEARCH_MAX_RESULTS,
        )
        all_results.extend(results)

    if SEARCH_MAX_TOTAL > 0:
        all_results = all_results[:SEARCH_MAX_TOTAL]
    return {
        "search_results": all_results,
        "search_plan": [],
        "planner": {},
        "subquestions": [],
    }


async def extract_node(state: GraphState) -> Dict[str, Any]:
    search_blob = compact_search_results(
        state.get("search_results", []) or [],
        max_items=EXTRACT_MAX_ITEMS,
        max_snippet_chars=EXTRACT_SNIPPET_CHARS,
    )
    prompt = REASONER_PROMPT.invoke(
        {
            "question": state.get("question", ""),
            "context": state.get("context", ""),
            "search_results": search_blob,
        }
    )
    resp = await extract_structured.ainvoke(prompt)
    data = resp if isinstance(resp, ReasonerOutput) else ReasonerOutput.model_validate(resp)
    context_update = data.context_update.strip()
    should_continue = bool(data.should_continue)
    draft_final = data.draft_final_answer.strip()
    iterations = int(state.get("iterations", 0)) + 1
    return {
        "accepted_facts": [_dump_model(fact) for fact in data.accepted_facts],
        "rejected_or_uncertain": [_dump_model(item) for item in data.rejected_or_uncertain],
        "context": context_update,
        "should_continue": should_continue,
        "draft_final_answer": draft_final,
        "iterations": iterations,
        "search_results": [],
    }


async def finalize_node(state: GraphState) -> Dict[str, Any]:
    draft = state.get("draft_final_answer", "") or ""
    if not draft:
        prompt = FINAL_PROMPT.invoke(
            {
                "question": state.get("question", ""),
                "context": state.get("context", ""),
            }
        )
        resp = await final_llm.ainvoke(prompt)
        draft = str(resp.content).strip()
    return {"answer": normalize_answer(draft)}


def _route_after_extract(state: GraphState) -> str:
    if state.get("should_continue", True) and int(state.get("iterations", 0)) < MAX_ITERS:
        return "continue"
    return "final"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("prep", prep_node)
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "final" if state.get("draft_final_answer") else "continue",
        {"continue": "prep", "final": "finalize"},
    )
    graph.add_edge("prep", "search")
    graph.add_edge("search", "extract")
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"continue": "planner", "final": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


_GRAPH = build_graph()


async def solve_async(question: str) -> str:
    result = await _GRAPH.ainvoke({"question": question, "context": "", "iterations": 0})
    return result.get("answer", "")


def solve(question: str) -> str:
    return asyncio.run(solve_async(question))
