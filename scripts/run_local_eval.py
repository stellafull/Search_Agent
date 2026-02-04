from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import solve  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_PATH = DATA_DIR / "question.jsonl"
OUTPUT_PATH = DATA_DIR / "answer.jsonl"
# Concurrent workers - sliding window mode keeps all workers busy
EVAL_WORKERS = int(os.getenv("EVAL_WORKERS", "4"))
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("EVAL_RETRY_DELAY", "60.0"))


def solve_with_retry(question: str, max_retries: int = MAX_RETRIES) -> str:
    """Solve with retry logic for connection errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return solve(question)
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            # Retry on connection/network errors
            if "connection" in err_str or "timeout" in err_str or "rate" in err_str:
                delay = RETRY_DELAY * (2 ** attempt)  # exponential backoff
                print(f"[Retry {attempt+1}/{max_retries}] Connection error, waiting {delay}s...")
                time.sleep(delay)
                continue
            # Don't retry on other errors
            raise
    # All retries exhausted
    raise last_error


def load_questions(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    questions = load_questions(QUESTIONS_PATH)
    results: List[Dict] = [None] * len(questions)

    error_log = DATA_DIR / "answer_errors.log"
    if error_log.exists():
        error_log.unlink()

    total = len(questions)
    completed = 0
    
    print(f"Starting evaluation: {total} questions, {EVAL_WORKERS} concurrent workers")
    
    # Submit all tasks at once - ThreadPoolExecutor manages concurrency automatically
    # This keeps workers fully utilized (sliding window instead of batch-wait)
    with futures.ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        future_map = {
            executor.submit(solve_with_retry, row["question"]): idx
            for idx, row in enumerate(questions)
        }
        
        for fut in futures.as_completed(future_map):
            idx = future_map[fut]
            row = questions[idx]
            try:
                answer = fut.result()
                completed += 1
                print(f"[OK] id={row['id']} ({completed}/{total})")
            except Exception as exc:
                answer = ""
                completed += 1
                print(f"[FAIL] id={row['id']} ({completed}/{total}): {exc!r}")
                with error_log.open("a", encoding="utf-8") as logf:
                    logf.write(f"id={row['id']} error={exc!r}\n")
            results[idx] = {"id": row["id"], "answer": answer}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
