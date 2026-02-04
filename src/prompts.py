from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.state import Evidence


class GapAnalysisResult(BaseModel):
    current_gap: str = Field(
        ..., description="Single most critical missing need, or READY_TO_ANSWER"
    )
    reason: str = Field(..., description="Short rationale for why this gap matters")


class QueryPlanResult(BaseModel):
    queries: List[str] = Field(..., description="1-2 precise search queries")
    tool_hint: Optional[Literal["tavily", "iqs", "both"]] = Field(
        default="both", description="Preferred tool(s) for retrieval"
    )
    rationale: str = Field(..., description="Why these queries should close the gap")


class CritiqueResult(BaseModel):
    verdict: Literal["PASS", "WEAK", "FAIL"] = Field(
        ..., description="PASS if explicit evidence supports the gap"
    )
    extracted_facts: List[Evidence] = Field(
        default_factory=list,
        description="Only facts explicitly supported by search results",
    )
    missing: Optional[str] = Field(
        default=None, description="What is still missing if not PASS"
    )
    retry_hint: Optional[str] = Field(
        default=None, description="Actionable next search hint"
    )


class ReflectionResult(BaseModel):
    reflection: str = Field(..., description="Short strategy adjustment")
    ttl_hops: int = Field(
        default=2, description="Apply this reflection for the next N hops"
    )


class FinalAnswerResult(BaseModel):
    answer: str = Field(..., description="Final answer string only")


GAP_ANALYSIS_PROMPT = """You are the Gap Analyzer for a multi-hop research agent.
Identify the single most critical missing need to answer the question.
If facts are sufficient, return READY_TO_ANSWER.

Rules:
- Carefully identify WHAT the question is asking for.
- Focus on exactly ONE gap (the next missing fact).
- Use known facts only; do not invent.
- If reflection provided, change strategy.
- Be concise in your reason (1-2 sentences max).
"""

QUERY_PLAN_PROMPT = """You are the Query Planner.
Convert the current_gap into 1-2 precise search queries.

Rules:
- Use KEYWORD style, NOT full sentences
- Keep queries short (<80 chars) and specific.
- Never start queries with "The", "Find", "What", "Search for".
- **LANGUAGE STRATEGY**:
  - For Chinese entities → use Chinese keywords
  - For English entities → use English keywords  
  - For Russian/Arabic/other languages → use ENGLISH keywords (more searchable)
  - If reflection suggests a specific language, follow it only if reasonable.
- Include key entities, proper nouns, or technical terms.
- Apply reflection hints if provided.
- Keep rationale brief (1 sentence).
"""

CRITIQUE_PROMPT = """You are the Fact Extractor and Critic.
Given the current_gap and search results, extract relevant facts and judge confidence level.

Rules:
- Extract ANY facts from search results that are relevant to the question or gap, even if partially supported.
- PASS: Clear, explicit evidence directly answers the gap.
- WEAK: Some relevant information found but not fully conclusive - still extract the candidate facts.
- FAIL: No relevant information at all.
- Each extracted fact should be a concise statement (no URLs or snippets needed).
- IMPORTANT: Even for WEAK verdict, you MUST extract candidate facts if any relevant info exists.
- If not PASS, provide missing + retry_hint.
"""

REFLECTION_PROMPT = """You are the Reflexion module.
When the agent is stuck, propose a short, concrete strategy change for the next hops.

Rules:
- Actionable and concise.
- Only suggest STRATEGY changes (language, keywords, search angle).
- NEVER include specific entity names or answer candidates that you are not certain about.
- Examples: 
  - Switch query language
  - Add constraint keyword (year, location, attribute)
  - Search for a related entity type first
  - Try different search angles or synonyms
"""

FINAL_ANSWER_PROMPT = """You are the Final Answer module.
Based on the question, confirmed facts, and search context, output the final answer.

Rules:
- **CRITICAL: Re-read the question carefully. Answer EXACTLY what is asked.**
  - If it asks for "名称" (name), give the name only.
- **ANSWER LANGUAGE**: 
  - If the question specifies a language (e.g., "用中文回答", "in English"), use that language.
  - Otherwise, use the entity's native/official name (e.g., Russian company → Russian name, Chinese game → Chinese name).
- If confirmed facts contain the answer, use them directly.
- If confirmed facts are empty but search context is provided, carefully analyze the search results to extract the answer.
- Return ONLY the answer string; be as concise as possible.
- Do not explain or add extra text.
"""
