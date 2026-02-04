from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.state import Evidence, SearchResultItem, ToolStep

_ = load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "200"))  # Allow longer queries for complex topics
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "6"))
MAX_RESULTS_TOTAL = int(os.getenv("MAX_RESULTS_TOTAL", "18"))
SIMILARITY_THRESHOLD = float(os.getenv("QUERY_SIM_THRESHOLD", "0.9"))
IQS_DISABLED = os.getenv("IQS_DISABLED", "1").lower() not in {"0", "false", "no"}

# ========= LLM factory =========


def build_llm(
    model: str, temperature: float = 0.0, max_tokens: Optional[int] = None
):
    if max_tokens is None:
        try:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        except ValueError:
            max_tokens = 512
    kwargs = {
        "model": model,
        "model_provider": os.getenv("LLM_PROVIDER", "openai"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)


# ========= Text helpers =========


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate_text(text: str, max_len: int = 240) -> str:
    if text is None:
        return ""
    text = normalize_whitespace(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = normalize_whitespace(text)
    text = text.casefold()
    # Remove unicode punctuation and symbols, keep letters/numbers/whitespace/CJK
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        cleaned.append(ch)
    text = "".join(cleaned)
    text = normalize_whitespace(text)
    return text


# ========= Query guardrails =========


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def validate_query(
    query: str,
    previous_queries: Sequence[str],
    current_gap: Optional[str] = None,
) -> Tuple[bool, str]:
    query = normalize_whitespace(query)
    if not query:
        return False, "empty"
    if len(query) > MAX_QUERY_LEN:
        return False, "too_long"
    for prev in previous_queries:
        if _similarity(query, prev) >= SIMILARITY_THRESHOLD:
            return False, "too_similar"
    if current_gap:
        quoted_terms = re.findall(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]", current_gap)
        if quoted_terms:
            if not any(term in query for term in quoted_terms):
                return False, "missing_quoted_terms"
    return True, "ok"


# Common filler words that hurt search quality when at the start
_QUERY_STRIP_PREFIXES = re.compile(
    r"^(the|a|an|this|that|what|which|find|search|identify|locate|determine)\s+",
    re.IGNORECASE
)


def rewrite_query(query: str, current_gap: Optional[str] = None) -> str:
    query = normalize_whitespace(query)
    
    # Remove filler prefixes that hurt search quality
    # e.g., "The specific paper title..." -> "specific paper title..."
    query = _QUERY_STRIP_PREFIXES.sub("", query)
    query = normalize_whitespace(query)
    
    # Remove brackets and parentheses
    query = re.sub(r"[()\[\]{}]", " ", query)
    query = normalize_whitespace(query)
    
    # Truncate at word boundary if too long
    if len(query) > MAX_QUERY_LEN:
        truncated = query[:MAX_QUERY_LEN]
        last_space = truncated.rfind(" ")
        if last_space > MAX_QUERY_LEN // 2:  # Only if we keep at least half
            query = truncated[:last_space]
        else:
            query = truncated
    
    query = normalize_whitespace(query)
    if current_gap and len(query) < 6:
        query = normalize_whitespace(f"{current_gap} {query}")
    return query


def dedup_queries(queries: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in queries:
        qn = normalize_whitespace(q)
        if not qn or qn in seen:
            continue
        seen.add(qn)
        out.append(qn)
    return out


# ========= Fact/rendering helpers =========

# Context budget controls - tune these to balance completeness vs token efficiency
MAX_FACTS_FOR_LLM = int(os.getenv("MAX_FACTS_FOR_LLM", "8"))  # Most recent facts


def format_facts(facts: Sequence[Evidence], max_facts: int = MAX_FACTS_FOR_LLM) -> str:
    """Format known facts for LLM consumption, limiting context size."""
    if not facts:
        return "(none)"
    lines = []
    # Show most recent facts first (more relevant)
    display_facts = facts[-max_facts:] if len(facts) > max_facts else facts
    for idx, fact in enumerate(display_facts, start=1):
        # Compact format: just the fact (no source info to save tokens)
        lines.append(f"{idx}. {fact.fact}")
    if len(facts) > max_facts:
        lines.insert(0, f"(showing {max_facts} of {len(facts)} facts)")
    return "\n".join(lines)


# Default limits for search results - can be overridden per-call
MAX_RESULTS_FOR_LLM = int(os.getenv("MAX_RESULTS_FOR_LLM", "6"))
MAX_SNIPPET_LEN = int(os.getenv("MAX_SNIPPET_LEN", "150"))


def format_search_results(
    results: Sequence[SearchResultItem],
    max_results: int = MAX_RESULTS_FOR_LLM,
    max_snippet_len: int = MAX_SNIPPET_LEN,
) -> str:
    """Format search results for LLM consumption, limiting context size."""
    if not results:
        return "(none)"
    lines = []
    for idx, item in enumerate(results[:max_results], start=1):
        title = truncate_text(item.title or "", 60)
        snippet = truncate_text(item.snippet or "", max_snippet_len)
        # Omit URL to save tokens - source is enough for credibility
        source = item.source or "search"
        lines.append(f"{idx}. [{source}] {title}: {snippet}")
    if len(results) > max_results:
        lines.append(f"... ({len(results) - max_results} more results omitted)")
    return "\n".join(lines)


# ========= Search tools (MCP + direct) =========

_TAVILY_TOOLS: Optional[list] = None
_IQS_TOOLS: Optional[list] = None


async def get_tavily_mcp_tools():
    global _TAVILY_TOOLS
    if _TAVILY_TOOLS is not None:
        return _TAVILY_TOOLS
    try:
        client = MultiServerMCPClient(
            {
                "tavily-remote-mcp": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": [
                        "-y",
                        "mcp-remote",
                        f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}",
                    ],
                    "env": {},
                }
            }
        )
        _TAVILY_TOOLS = await client.get_tools(server_name="tavily-remote-mcp")
    except Exception as exc:
        logger.warning("tavily mcp init failed: %s", exc)
        _TAVILY_TOOLS = []
    return _TAVILY_TOOLS


async def get_iqs_mcp_tools():
    if IQS_DISABLED:
        return []
    global _IQS_TOOLS
    if _IQS_TOOLS is not None:
        return _IQS_TOOLS
    try:
        client = MultiServerMCPClient(
            {
                "iqs-mcp-server-maps": {
                    "transport": "streamable_http",
                    "url": "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-search",
                    "headers": {
                        "X-API-Key": os.getenv("ALIBABA_IQS_API_KEY"),
                    },
                }
            }
        )
        _IQS_TOOLS = await client.get_tools(server_name="iqs-mcp-server-maps")
    except Exception as exc:
        logger.warning("iqs mcp init failed: %s", exc)
        _IQS_TOOLS = []
    return _IQS_TOOLS


def _extract_fields_from_schema(tool: Any) -> List[str]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return []
    if hasattr(args_schema, "model_fields"):
        return list(args_schema.model_fields.keys())
    if hasattr(args_schema, "__fields__"):
        return list(args_schema.__fields__.keys())
    return []


def _build_tool_payload(tool: Any, query: str, max_results: int) -> Dict[str, Any]:
    fields = _extract_fields_from_schema(tool)
    payload: Dict[str, Any] = {}
    if fields:
        for name in fields:
            lname = name.lower()
            if lname in {"query", "q", "text", "input", "keyword", "keywords"}:
                payload[name] = query
            elif lname in {
                "k",
                "limit",
                "top_k",
                "max_results",
                "num_results",
                "count",
            }:
                payload[name] = max_results
    if not payload:
        payload = {"query": query}
    return payload


def _parse_search_output(output: Any, source: str) -> List[SearchResultItem]:
    if output is None:
        return []
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return [SearchResultItem(title="", snippet=output, source=source)]
    data: Any = output
    if isinstance(output, dict):
        for key in ("results", "items", "data", "documents", "docs"):
            if key in output:
                data = output[key]
                break
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    items: List[SearchResultItem] = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            items.append(
                SearchResultItem(
                    title="",
                    snippet=item,
                    source=source,
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or ""
        snippet = (
            item.get("content")
            or item.get("snippet")
            or item.get("text")
            or item.get("description")
            or ""
        )
        items.append(
            SearchResultItem(
                title=title,
                snippet=snippet,
                source=source,
            )
        )
    return items


async def _search_with_mcp_tools(
    tools: Sequence[Any],
    query: str,
    source: str,
    max_results: int,
) -> List[SearchResultItem]:
    if not tools:
        return []
    # prefer tools with search-like names
    ordered = sorted(tools, key=lambda t: "search" not in getattr(t, "name", ""))
    for tool in ordered:
        payload = _build_tool_payload(tool, query, max_results)
        try:
            output = await tool.ainvoke(payload)
        except Exception:
            continue
        items = _parse_search_output(output, source)
        if items:
            return items[:max_results]
    return []


SEARCH_MAX_RETRIES = int(os.getenv("SEARCH_MAX_RETRIES", "2"))
SEARCH_RETRY_DELAY = float(os.getenv("SEARCH_RETRY_DELAY", "2.0"))


async def search_tavily(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[SearchResultItem]:
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        for attempt in range(SEARCH_MAX_RETRIES + 1):
            try:
                from tavily import TavilyClient

                client = TavilyClient(api_key=api_key)
                resp = client.search(
                    query=query,
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=False,
                )
                results = resp.get("results") if isinstance(resp, dict) else None
                if results:
                    return _parse_search_output({"results": results}, "tavily")
                break  # No results but no error, don't retry
            except Exception as exc:
                err_str = str(exc).lower()
                if attempt < SEARCH_MAX_RETRIES and ("connection" in err_str or "timeout" in err_str):
                    logger.warning("tavily search retry %d: %s", attempt + 1, exc)
                    await asyncio.sleep(SEARCH_RETRY_DELAY * (attempt + 1))
                    continue
                logger.warning("tavily direct search failed: %s", exc)
                break
    tools = await get_tavily_mcp_tools()
    return await _search_with_mcp_tools(tools, query, "tavily", max_results)


async def search_iqs(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[SearchResultItem]:
    if IQS_DISABLED:
        return []
    tools = await get_iqs_mcp_tools()
    return await _search_with_mcp_tools(tools, query, "iqs", max_results)


def merge_dedup_results(results: Iterable[SearchResultItem]) -> List[SearchResultItem]:
    seen = set()
    merged: List[SearchResultItem] = []
    for item in results:
        # Dedup by title + snippet prefix (URL removed)
        snippet_prefix = item.snippet[:50] if item.snippet else ""
        key = f"{item.title}|{snippet_prefix}"
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= MAX_RESULTS_TOTAL:
            break
    return merged


def build_tool_step(tool: str, query: str, num_results: int, notes: str | None = None) -> ToolStep:
    return ToolStep(tool=tool, query=query, num_results=num_results, notes=notes)


async def run_parallel_searches(
    query: str, use_tavily: bool, use_iqs: bool
) -> List[SearchResultItem]:
    tasks = []
    if use_tavily:
        tasks.append(search_tavily(query))
    if use_iqs:
        tasks.append(search_iqs(query))
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged: List[SearchResultItem] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        merged.extend(res)
    return merged
