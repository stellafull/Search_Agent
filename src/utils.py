from __future__ import annotations

import ast
import json
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

_ = load_dotenv(find_dotenv())

# ========= LLMs =========

extract_model = init_chat_model(
    model=os.getenv("EXTRACT_MODEL", "Qwen/Qwen3-VL-8B-Thinking"),
    model_provider=os.getenv("LLM_PROVIDER", "openai"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0,
)


# ========= JSON helpers =========

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def safe_json_loads(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if text is None:
        return {}
    raw = str(text).strip()
    if not raw:
        return {}
    raw = _strip_code_fence(raw)
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
        try:
            return ast.literal_eval(snippet)
        except Exception:
            pass

    fixed = raw.replace("\n", " ").replace("\t", " ")
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    try:
        return json.loads(fixed)
    except Exception:
        return {}


# ========= Normalization =========


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""
    cleaned = []
    for ch in text:
        if ch.isspace():
            cleaned.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        cleaned.append(ch)
    normalized = re.sub(r"\s+", " ", "".join(cleaned)).strip()
    return normalized


# ========= Search tools (MCP) =========

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
    except Exception:
        _TAVILY_TOOLS = []
    return _TAVILY_TOOLS


async def get_iqs_mcp_tools():
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
    except Exception:
        _IQS_TOOLS = []
    return _IQS_TOOLS


def _pick_search_tool(tools: list) -> Any:
    if not tools:
        return None
    for tool in tools:
        if "search" in tool.name.lower():
            return tool
    return tools[0]


def _build_tool_input(tool: Any, query: str, max_results: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    schema = getattr(tool, "args_schema", None)
    fields = set()
    if schema is not None and hasattr(schema, "model_fields"):
        fields = set(schema.model_fields.keys())
    if "query" in fields:
        payload["query"] = query
    elif "q" in fields:
        payload["q"] = query
    elif "search_query" in fields:
        payload["search_query"] = query
    else:
        payload["query"] = query
    if "max_results" in fields:
        payload["max_results"] = max_results
    elif "limit" in fields:
        payload["limit"] = max_results
    return payload


def _normalize_results(raw: Any, source: str, query: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def _add_items(items: List[Dict[str, Any]]):
        for item in items:
            add_item(item)

    def _parse_markdown_results(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not text:
            return items
        blocks = re.split(r"\n---\n", text)
        for block in blocks:
            title_match = re.search(r"##\s*标题\s*(.+)", block)
            if not title_match:
                continue
            title = title_match.group(1).strip()
            url_match = re.search(r"\*\*url\*\*:\s*(\S+)", block)
            snippet_match = re.search(r"\*\*摘要\*\*:\s*(.+)", block)
            url = url_match.group(1).strip() if url_match else ""
            snippet = snippet_match.group(1).strip() if snippet_match else ""
            items.append({"title": title, "url": url, "snippet": snippet})
        return items

    def add_item(item: Dict[str, Any]):
        results.append(
            {
                "source": source,
                "query": query,
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", item.get("content", ""))),
                "raw": item,
            }
        )

    if isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], str):
            text = raw["text"].strip()
            if text.startswith("{") and "results" in text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "results" in parsed:
                        for item in parsed["results"]:
                            if isinstance(item, dict):
                                add_item(item)
                        return results
                except Exception:
                    pass
            parsed_items = _parse_markdown_results(text)
            if parsed_items:
                _add_items(parsed_items)
                return results
        if "results" in raw and isinstance(raw["results"], list):
            for item in raw["results"]:
                if isinstance(item, dict):
                    add_item(item)
            return results
        if "data" in raw and isinstance(raw["data"], list):
            for item in raw["data"]:
                if isinstance(item, dict):
                    add_item(item)
            return results
        if "items" in raw and isinstance(raw["items"], list):
            for item in raw["items"]:
                if isinstance(item, dict):
                    add_item(item)
            return results
        add_item(raw)
        return results
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                add_item(item)
            else:
                add_item({"snippet": str(item)})
        return results
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") and "results" in text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "results" in parsed:
                    for item in parsed["results"]:
                        if isinstance(item, dict):
                            add_item(item)
                    return results
            except Exception:
                pass
        parsed_items = _parse_markdown_results(text)
        if parsed_items:
            _add_items(parsed_items)
            return results
    add_item({"snippet": str(raw)})
    return results


async def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    tools = await get_tavily_mcp_tools()
    tool = _pick_search_tool(tools)
    if tool is None:
        return []
    payload = _build_tool_input(tool, query, max_results)
    try:
        raw = await tool.ainvoke(payload)
    except Exception:
        return []
    return _normalize_results(raw, "tavily", query)


async def iqs_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    tools = await get_iqs_mcp_tools()
    tool = _pick_search_tool(tools)
    if tool is None:
        return []
    payload = _build_tool_input(tool, query, max_results)
    try:
        raw = await tool.ainvoke(payload)
    except Exception:
        return []
    return _normalize_results(raw, "iqs", query)


async def search_all(
    iqs_queries: Iterable[str],
    tavily_queries: Iterable[str],
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    tasks = []
    for q in iqs_queries:
        if q:
            tasks.append(iqs_search(q, max_results=max_results))
    for q in tavily_queries:
        if q:
            tasks.append(tavily_search(q, max_results=max_results))
    results: List[Dict[str, Any]] = []
    if not tasks:
        return results
    for batch in await _gather_with_concurrency(tasks):
        if isinstance(batch, list):
            results.extend(batch)
    return results


async def _gather_with_concurrency(tasks: List[Any], limit: int = 6) -> List[Any]:
    import asyncio

    semaphore = asyncio.Semaphore(limit)

    async def _run(task):
        async with semaphore:
            return await task

    results = await asyncio.gather(*[_run(t) for t in tasks], return_exceptions=True)
    cleaned: List[Any] = []
    for item in results:
        if isinstance(item, Exception):
            continue
        cleaned.append(item)
    return cleaned


def compact_search_results(
    results: List[Dict[str, Any]],
    max_items: int = 30,
    max_snippet_chars: int = 400,
) -> str:
    trimmed: List[Dict[str, Any]] = []
    for item in results[:max_items]:
        snippet = item.get("snippet", "")
        if isinstance(snippet, str) and max_snippet_chars > 0 and len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "..."
        trimmed.append(
            {
                "source": item.get("source", ""),
                "query": item.get("query", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": snippet,
            }
        )
    return json.dumps(trimmed, ensure_ascii=False, indent=2)


# ========= IO helpers =========


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
