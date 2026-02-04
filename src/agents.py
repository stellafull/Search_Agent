from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Dict, List

import openai
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.prompts import (
    CRITIQUE_PROMPT,
    FINAL_ANSWER_PROMPT,
    GAP_ANALYSIS_PROMPT,
    QUERY_PLAN_PROMPT,
    REFLECTION_PROMPT,
    CritiqueResult,
    FinalAnswerResult,
    GapAnalysisResult,
    QueryPlanResult,
    ReflectionResult,
)
from src.state import AgentState, SearchResultItem
from src.utils import (
    build_llm,
    build_tool_step,
    dedup_queries,
    format_facts,
    format_search_results,
    IQS_DISABLED,
    merge_dedup_results,
    normalize_answer,
    rewrite_query,
    run_parallel_searches,
    validate_query,
)

MAX_HOPS = int(os.getenv("MAX_HOPS", "6"))
NO_DELTA_THRESHOLD = int(os.getenv("NO_DELTA_THRESHOLD", "2"))
MAX_TOOL_CALLS_PER_HOP = int(os.getenv("MAX_TOOL_CALLS_PER_HOP", "2"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

# Use Thinking model for complex reasoning tasks
REASON_MODEL = os.getenv("REASON_MODEL", "Qwen/Qwen3-VL-235B-A22B-Thinking")
# Use non-Thinking instruct model for simple tasks (query generation) to avoid wasted tokens
FAST_MODEL = os.getenv("FAST_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

# Max tokens settings
REASON_MAX_TOKENS = _env_int("REASON_MAX_TOKENS", 8192)  # Thinking models need more room
FAST_MAX_TOKENS = _env_int("FAST_MAX_TOKENS", 1024)  # Non-thinking model needs less
FINAL_MAX_TOKENS = _env_int("FINAL_MAX_TOKENS", 512)

planner_llm = build_llm(REASON_MODEL, temperature=0, max_tokens=REASON_MAX_TOKENS).with_structured_output(
    GapAnalysisResult
)
query_llm = build_llm(FAST_MODEL, temperature=0, max_tokens=FAST_MAX_TOKENS).with_structured_output(QueryPlanResult)
critic_llm = build_llm(REASON_MODEL, temperature=0, max_tokens=REASON_MAX_TOKENS).with_structured_output(
    CritiqueResult
)
# Use REASON_MODEL for reflect to avoid length limit issues with Thinking models
reflect_llm = build_llm(REASON_MODEL, temperature=0, max_tokens=REASON_MAX_TOKENS).with_structured_output(
    ReflectionResult
)
final_llm = build_llm(REASON_MODEL, temperature=0, max_tokens=FINAL_MAX_TOKENS).with_structured_output(
    FinalAnswerResult
)


_gap_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GAP_ANALYSIS_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nKnown facts:\n{known_facts}\n\nPrevious gap:\n{last_gap}\n\nReflection:\n{reflection}\n",
        ),
    ]
)
_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_PLAN_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nCurrent gap:\n{current_gap}\n\nKnown facts:\n{known_facts}\n\nReflection:\n{reflection}\n",
        ),
    ]
)
_critique_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIQUE_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nCurrent gap:\n{current_gap}\n\nKnown facts:\n{known_facts}\n\nSearch results:\n{search_results}\n",
        ),
    ]
)
_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REFLECTION_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nCurrent gap:\n{current_gap}\n\nRecent queries:\n{recent_queries}\n\nKnown facts:\n{known_facts}\n\nNo-delta streak:\n{no_delta_streak}\n",
        ),
    ]
)
_final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FINAL_ANSWER_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nConfirmed facts:\n{known_facts}\n\nSearch context (for reference if facts are insufficient):\n{search_context}\n",
        ),
    ]
)

gap_chain = _gap_prompt | planner_llm
query_chain = _query_prompt | query_llm
critique_chain = _critique_prompt | critic_llm
reflection_chain = _reflection_prompt | reflect_llm
final_chain = _final_prompt | final_llm

_LENGTH_ERROR_TYPE = getattr(openai, "LengthFinishReasonError", None)


def _is_length_error(exc: Exception) -> bool:
    """Check if exception is caused by LLM output length limit."""
    seen = set()
    cur: Exception | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # Check by type
        if _LENGTH_ERROR_TYPE and isinstance(cur, _LENGTH_ERROR_TYPE):
            return True
        # Check by class name (more robust across versions)
        if type(cur).__name__ == "LengthFinishReasonError":
            return True
        # Check by message content
        msg = str(cur).lower()
        if "length limit" in msg or "lengthfinishreason" in msg or "max_tokens" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _safe_gap_fallback(state: AgentState) -> str:
    """Generate fallback gap when LLM fails."""
    known_count = len(state.get("known_facts", []))
    if known_count == 0:
        return "identify key entities mentioned in the question"
    return "find more details about the entities already discovered"


async def analyze_gap(state: AgentState) -> AgentState:
    known_facts = format_facts(state.get("known_facts", []))
    reflection_ttl = state.get("reflection_ttl", 0)
    reflection_text = state.get("reflection") if reflection_ttl > 0 else "(none)"
    last_gap = state.get("current_gap") or "(none)"

    try:
        result = await gap_chain.ainvoke(
            {
                "question": state.get("question", ""),
                "known_facts": known_facts,
                "reflection": reflection_text,
                "last_gap": last_gap,
            }
        )
        current_gap = result.current_gap.strip()
    except Exception as exc:
        if _is_length_error(exc):
            current_gap = _safe_gap_fallback(state)
        else:
            raise
    updates: AgentState = {
        "last_gap": state.get("current_gap"),
        "current_gap": current_gap,
        "loop_count": state.get("loop_count", 0) + 1,
    }

    if reflection_ttl > 0:
        new_ttl = max(reflection_ttl - 1, 0)
        updates["reflection_ttl"] = new_ttl
        if new_ttl == 0:
            updates["reflection"] = None

    return updates


async def build_query(state: AgentState) -> AgentState:
    known_facts = format_facts(state.get("known_facts", []))
    reflection = state.get("reflection") or "(none)"
    
    try:
        result = await query_chain.ainvoke(
            {
                "question": state.get("question", ""),
                "current_gap": state.get("current_gap", ""),
                "known_facts": known_facts,
                "reflection": reflection,
            }
        )
        raw_queries = dedup_queries(result.queries)
        tool_hint = result.tool_hint or "both"
    except Exception as exc:
        if _is_length_error(exc):
            # Fallback: use current_gap as query
            gap = state.get("current_gap") or state.get("question", "")
            raw_queries = [rewrite_query(gap, None)]
            tool_hint = "both"
        else:
            raise

    history = state.get("_query_history", [])
    validated: List[str] = []
    for query in raw_queries:
        ok, _reason = validate_query(query, history, state.get("current_gap"))
        if ok:
            validated.append(query)
        else:
            rewritten = rewrite_query(query, state.get("current_gap"))
            ok2, _reason2 = validate_query(rewritten, history, state.get("current_gap"))
            if ok2:
                validated.append(rewritten)

    if not validated:
        fallback_seed = state.get("current_gap") or state.get("question", "")
        fallback = rewrite_query(fallback_seed, None)
        if fallback:
            validated = [fallback]

    validated = dedup_queries(validated)[:3]
    # Limit query history to prevent unbounded state growth
    MAX_QUERY_HISTORY = 10
    new_history = history + [q for q in validated if q not in history]
    new_history = new_history[-MAX_QUERY_HISTORY:]  # Keep only recent queries

    return {
        "current_queries": validated,
        "_query_history": new_history,
        "current_tool_hint": tool_hint,
    }


async def search(state: AgentState) -> AgentState:
    queries = state.get("current_queries", [])
    tool_hint = (state.get("current_tool_hint") or "both").lower()
    use_tavily = tool_hint in {"tavily", "both"}
    use_iqs = (not IQS_DISABLED) and tool_hint in {"iqs", "both"}

    all_results: List[SearchResultItem] = []
    # _tool_steps: only keep current hop's steps (reset each hop to prevent unbounded growth)
    tool_steps = []

    for query in queries:
        results = await run_parallel_searches(query, use_tavily, use_iqs)
        all_results.extend(results)
        if use_tavily:
            tavily_count = len([r for r in results if r.source == "tavily"])
            tool_steps.append(build_tool_step("tavily", query, tavily_count))
        if use_iqs:
            iqs_count = len([r for r in results if r.source == "iqs"])
            tool_steps.append(build_tool_step("iqs", query, iqs_count))

        if not results and MAX_TOOL_CALLS_PER_HOP > 1:
            rewritten = rewrite_query(query, state.get("current_gap"))
            if rewritten and rewritten != query:
                retry_results = await run_parallel_searches(
                    rewritten, use_tavily, use_iqs
                )
                all_results.extend(retry_results)

    merged = merge_dedup_results(all_results)
    # search_results is replaced each hop (not accumulated)
    # _tool_steps is for debug only
    return {"search_results": merged, "_tool_steps": tool_steps}


async def extract_critique(state: AgentState) -> AgentState:
    known_facts = format_facts(state.get("known_facts", []))
    search_results = format_search_results(state.get("search_results", []))

    result = None  # Initialize to avoid UnboundLocalError
    try:
        result = await critique_chain.ainvoke(
            {
                "question": state.get("question", ""),
                "current_gap": state.get("current_gap", ""),
                "known_facts": known_facts,
                "search_results": search_results,
            }
        )
        extracted = list(result.extracted_facts or [])
        verdict = result.verdict
    except Exception as exc:
        if _is_length_error(exc):
            # Fallback: treat as WEAK, no new facts extracted
            extracted = []
            verdict = "WEAK"
        else:
            raise

    if verdict == "PASS" and not extracted:
        verdict = "WEAK"

    updates: AgentState = {"last_verdict": verdict}

    # CRITICAL FIX: Add extracted facts even for WEAK verdict
    # This ensures we accumulate partial knowledge instead of losing it
    if extracted:
        updates["known_facts"] = list(state.get("known_facts", [])) + extracted
    
    if verdict == "PASS":
        updates["reflection"] = None
        updates["reflection_ttl"] = 0
        updates["no_delta_streak"] = 0
        updates["weak_streak"] = 0  # Reset on PASS
    else:
        # Track consecutive WEAK verdicts (even with facts)
        updates["weak_streak"] = state.get("weak_streak", 0) + 1
        # Only increment no_delta_streak if we didn't extract any facts
        if not extracted:
            updates["no_delta_streak"] = state.get("no_delta_streak", 0) + 1
        else:
            # We got some facts, reset the streak
            updates["no_delta_streak"] = 0
        if result and result.retry_hint:
            updates["reflection"] = result.retry_hint
            updates["reflection_ttl"] = 1

    return updates


async def reflect(state: AgentState) -> AgentState:
    known_facts = format_facts(state.get("known_facts", []))
    recent_queries = "\n".join(state.get("current_queries", [])) or "(none)"

    try:
        result = await reflection_chain.ainvoke(
            {
                "question": state.get("question", ""),
                "current_gap": state.get("current_gap", ""),
                "recent_queries": recent_queries,
                "known_facts": known_facts,
                "no_delta_streak": state.get("no_delta_streak", 0),
            }
        )
    except Exception as exc:
        if _is_length_error(exc):
            return {
                "reflection": "尝试用中文关键词搜索；或换一个相关实体先搜索",
                "reflection_ttl": 2,
                "no_delta_streak": 0,
                "weak_streak": 0,
            }
        raise

    return {
        "reflection": result.reflection,
        "reflection_ttl": max(result.ttl_hops, 1),
        "no_delta_streak": 0,
        "weak_streak": 0,  # Reset after reflection
    }


async def finalize(state: AgentState) -> AgentState:
    facts_list = state.get("known_facts", [])
    known_facts = format_facts(facts_list)
    
    # === Tiered context strategy ===
    # Tier 1: If we have sufficient confirmed facts, minimal search context needed
    # Tier 2: If facts are sparse, include more search context
    # Tier 3: If no facts, maximize search context (but still bounded)
    
    search_results = state.get("search_results", [])
    
    if len(facts_list) >= 3:
        # Tier 1: Rich facts - minimal search context (just for verification)
        search_context = format_search_results(search_results, max_results=3, max_snippet_len=100)
    elif len(facts_list) >= 1:
        # Tier 2: Some facts - moderate search context
        search_context = format_search_results(search_results, max_results=6, max_snippet_len=150)
    else:
        # Tier 3: No facts - need search context to infer answer
        # But still bounded to avoid context overflow
        search_context = format_search_results(search_results, max_results=8, max_snippet_len=180)
    
    # Only add query hints if truly no other context
    if search_context == "(none)" and known_facts == "(none)":
        query_history = state.get("_query_history", [])
        if query_history:
            # Just last 3 queries as hints
            search_context = f"(Queries tried: {', '.join(query_history[-3:])})"
    
    try:
        result = await final_chain.ainvoke(
            {
                "question": state.get("question", ""),
                "known_facts": known_facts,
                "search_context": search_context,
            }
        )
        answer = normalize_answer(result.answer)
    except Exception as exc:
        if _is_length_error(exc):
            # Fallback: extract answer from known facts if possible
            if facts_list:
                # Use the last fact as a best-effort answer
                answer = normalize_answer(facts_list[-1].fact)
            else:
                answer = ""
        else:
            raise
    
    return {"final_answer": answer}


def _route_after_gap(state: AgentState) -> str:
    if state.get("current_gap", "").strip().upper() == "READY_TO_ANSWER":
        return "finalize"
    return "build_query"


def _route_after_critique(state: AgentState) -> str:
    if state.get("loop_count", 0) >= MAX_HOPS:
        return "finalize"
    if state.get("last_verdict") == "PASS":
        return "analyze_gap"
    # Trigger reflect if stuck in WEAK loop (even with partial facts)
    # This helps when strategy is wrong (e.g., wrong language, wrong entity)
    WEAK_STREAK_THRESHOLD = 3
    if state.get("weak_streak", 0) >= WEAK_STREAK_THRESHOLD:
        return "reflect"
    # WEAK with new facts extracted -> go back to planner to reassess gap
    # This prevents infinite retry loops when making partial progress
    # (no_delta_streak == 0 means we just extracted some facts in this hop)
    if state.get("last_verdict") == "WEAK" and state.get("no_delta_streak", 0) == 0:
        return "analyze_gap"
    # FAIL or WEAK without new facts -> retry or reflect
    if state.get("no_delta_streak", 0) >= NO_DELTA_THRESHOLD:
        return "reflect"
    return "build_query"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyze_gap", analyze_gap)
    graph.add_node("build_query", build_query)
    graph.add_node("search", search)
    graph.add_node("extract_critique", extract_critique)
    graph.add_node("reflect", reflect)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("analyze_gap")

    graph.add_conditional_edges(
        "analyze_gap", _route_after_gap, {"build_query": "build_query", "finalize": "finalize"}
    )
    graph.add_edge("build_query", "search")
    graph.add_edge("search", "extract_critique")
    graph.add_conditional_edges(
        "extract_critique",
        _route_after_critique,
        {
            "analyze_gap": "analyze_gap",
            "build_query": "build_query",
            "reflect": "reflect",
            "finalize": "finalize",
        },
    )
    graph.add_edge("reflect", "analyze_gap")
    graph.add_edge("finalize", END)

    return graph.compile()


_graph = build_graph()


def _run_async(coro_func, *args, **kwargs):
    """Run an async function safely from sync context.
    
    Args:
        coro_func: An async function (not a coroutine object)
        *args, **kwargs: Arguments to pass to the async function
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - we can use asyncio.run directly
        return asyncio.run(coro_func(*args, **kwargs))

    # There's a running loop (e.g., in Jupyter or nested async context)
    # Run in a separate thread with its own event loop
    result_holder: Dict[str, Any] = {"result": None, "error": None}

    def runner():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result_holder["result"] = new_loop.run_until_complete(coro_func(*args, **kwargs))
        except Exception as e:
            result_holder["error"] = e
        finally:
            new_loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    
    if result_holder["error"] is not None:
        raise result_holder["error"]
    return result_holder["result"]


async def solve_async(question: str) -> str:
    initial_state: AgentState = {
        "question": question,
        # Accumulated knowledge
        "known_facts": [],
        # Current hop state (reset each hop)
        "current_gap": "",
        "current_queries": [],
        "current_tool_hint": None,
        "search_results": [],  # Only current hop results
        # Control flow
        "last_gap": None,
        "last_verdict": None,
        "reflection": None,
        "reflection_ttl": 0,
        "loop_count": 0,
        "no_delta_streak": 0,
        "weak_streak": 0,  # Track consecutive WEAK verdicts
        # Internal tracking (NOT passed to LLM)
        "_query_history": [],  # For dedup only
        "_tool_steps": [],  # Debug log only
        # Output
        "final_answer": None,
    }
    try:
        result = await _graph.ainvoke(initial_state)
        return result.get("final_answer", "")
    except Exception as exc:
        # Last resort fallback for any unhandled errors
        if _is_length_error(exc):
            return ""
        raise


def solve(question: str) -> str:
    """Synchronous entry point for solving a question."""
    return _run_async(solve_async, question)
