from __future__ import annotations

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """A search result item (URL omitted to save tokens)."""
    title: str = ""
    snippet: str = ""
    source: str = ""


class Evidence(BaseModel):
    """A verified fact extracted from search results."""
    fact: str = ""


class ToolStep(BaseModel):
    tool: str
    query: str
    num_results: int
    notes: Optional[str] = None


class FailureEvent(BaseModel):
    reason: str
    detail: Optional[str] = None


class AgentState(TypedDict, total=False):
    # === Core input ===
    question: str

    # === Accumulated knowledge (used in prompts) ===
    known_facts: List[Evidence]  # Confirmed facts with evidence

    # === Current hop state (reset each hop, used in prompts) ===
    current_gap: str  # Current gap being addressed
    current_queries: List[str]  # Queries for this hop
    current_tool_hint: Optional[str]  # Tool hint for this hop
    search_results: List[SearchResultItem]  # Results for THIS hop only (not accumulated)

    # === Control flow state ===
    last_gap: Optional[str]  # Previous gap (for detecting stuck)
    last_verdict: Optional[str]  # PASS/WEAK/FAIL from last critique
    reflection: Optional[str]  # Active reflection strategy
    reflection_ttl: int  # Hops remaining for reflection
    loop_count: int  # Total hops executed
    no_delta_streak: int  # Consecutive hops without new facts
    weak_streak: int  # Consecutive WEAK verdicts (even with facts) - triggers reflect if too high

    # === Internal tracking (NOT passed to LLM prompts) ===
    _query_history: List[str]  # All queries ever made (for dedup only)
    _tool_steps: List[ToolStep]  # Debug log (not sent to LLM)

    # === Output ===
    final_answer: Optional[str]
