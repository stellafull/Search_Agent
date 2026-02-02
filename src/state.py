from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from pydantic import BaseModel, Field


def merge_lists(a: List[Any] | None, b: List[Any] | None) -> List[Any]:
    return (a or []) + (b or [])


def merge_context(a: str | None, b: str | None) -> str:
    if not a:
        return b or ""
    if not b:
        return a
    if b in a:
        return a
    return f"{a}\n{b}"


class SubQuestion(TypedDict, total=False):
    id: str
    question: str
    rationale: str
    expected_evidence: str


class SubQuestionSchema(BaseModel):
    id: str = "SQ1"
    question: str = ""
    rationale: str = ""
    expected_evidence: str = ""

    model_config = {"extra": "ignore"}


class PlannerOutput(BaseModel):
    status: str = "search"
    subquestions: List[SubQuestionSchema] = Field(default_factory=list)
    final_answer: str = ""

    model_config = {"extra": "ignore"}


class SearchPlanItem(TypedDict, total=False):
    subquestion_id: str
    subquestion: str
    iqs_queries: List[str]
    tavily_queries: List[str]
    notes: str


class QueryGenOutput(BaseModel):
    lang: str = "mix"
    iqs_queries: List[str] = Field(default_factory=list)
    tavily_queries: List[str] = Field(default_factory=list)
    notes: str = ""

    model_config = {"extra": "ignore"}


class SearchResult(TypedDict, total=False):
    source: str
    query: str
    title: str
    url: str
    snippet: str
    raw: Any


class FactItem(TypedDict, total=False):
    fact: str
    why_accepted: str
    evidence: List[Dict[str, str]]


class EvidenceSchema(BaseModel):
    source: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""

    model_config = {"extra": "ignore"}


class FactItemSchema(BaseModel):
    fact: str = ""
    why_accepted: str = ""
    evidence: List[EvidenceSchema] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class RejectedItem(TypedDict, total=False):
    claim: str
    issue: str
    next_step_hint: str


class RejectedItemSchema(BaseModel):
    claim: str = ""
    issue: str = ""
    next_step_hint: str = ""

    model_config = {"extra": "ignore"}


class ReasonerOutput(BaseModel):
    accepted_facts: List[FactItemSchema] = Field(default_factory=list)
    rejected_or_uncertain: List[RejectedItemSchema] = Field(default_factory=list)
    context_update: str = ""
    should_continue: bool = True
    next_subquestions_hint: List[str] = Field(default_factory=list)
    draft_final_answer: str = ""

    model_config = {"extra": "ignore"}


class GraphState(TypedDict, total=False):
    question: str
    context: Annotated[str, merge_context]
    planner: Dict[str, Any]
    subquestions: List[SubQuestion]
    search_plan: List[SearchPlanItem]
    search_results: List[SearchResult]
    accepted_facts: Annotated[List[FactItem], merge_lists]
    rejected_or_uncertain: Annotated[List[RejectedItem], merge_lists]
    should_continue: bool
    draft_final_answer: str
    answer: str
    iterations: int
