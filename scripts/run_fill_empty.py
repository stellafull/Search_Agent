"""Retry questions with empty answers in answer.jsonl (multi-threaded)"""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import solve  # noqa: E402

DATA_DIR = ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "question.jsonl"
ANSWERS_PATH = DATA_DIR / "answer.jsonl"
ERRORS_LOG = DATA_DIR / "answer_errors.log"

MAX_RETRIES = int(os.getenv("RETRY_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "5.0"))
FILL_WORKERS = int(os.getenv("FILL_WORKERS", "5"))


def load_questions(path: Path) -> Dict[str, Dict]:
    """Load questions indexed by ID."""
    questions = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions[str(row["id"])] = row
    return questions


def load_answers_find_empty(path: Path) -> tuple[List[Dict], Set[str]]:
    """Load answers and find IDs with empty answers."""
    answers = []
    empty_ids = set()
    
    if not path.exists():
        return answers, empty_ids
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                answers.append(row)
                # Check for empty or missing answer (handles "", None, whitespace-only)
                answer = row.get("answer")
                if answer is None or (isinstance(answer, str) and not answer.strip()):
                    empty_ids.add(str(row["id"]))
            except json.JSONDecodeError:
                # Handle malformed JSON
                import re
                match = re.search(r'"id":\s*(\d+)', line)
                if match:
                    qid = match.group(1)
                    answers.append({"id": int(qid), "answer": ""})
                    empty_ids.add(qid)
    
    return answers, empty_ids


def solve_with_retry(question: str, max_retries: int = MAX_RETRIES) -> str:
    """Solve with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return solve(question)
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            # Retry on transient errors
            if any(kw in err_str for kw in ["connection", "timeout", "rate", "length"]):
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"  [Retry {attempt+1}/{max_retries}] Error: {type(exc).__name__}, waiting {delay}s...")
                time.sleep(delay)
                continue
            raise
    raise last_error


def main() -> None:
    # Load answers and find empty ones
    answers, empty_ids = load_answers_find_empty(ANSWERS_PATH)
    
    if not empty_ids:
        print("No empty answers found!")
        return
    
    print(f"Found {len(empty_ids)} empty answers to fill: {sorted(empty_ids, key=int)}")
    print(f"Using {FILL_WORKERS} concurrent workers")
    
    # Load questions
    questions = load_questions(QUESTIONS_PATH)
    
    # Build answer index by ID
    answer_index = {str(a["id"]): idx for idx, a in enumerate(answers)}
    
    # Thread-safe counters and results
    results_lock = threading.Lock()
    success_count = 0
    still_failed = []
    completed = 0
    total = len(empty_ids)
    
    def process_question(qid: str) -> tuple[str, str, Exception | None]:
        """Process a single question, return (qid, answer, error)."""
        if qid not in questions:
            return qid, "", ValueError(f"not found in questions")
        
        question = questions[qid]["question"]
        try:
            answer = solve_with_retry(question)
            return qid, answer, None
        except Exception as exc:
            return qid, "", exc
    
    # Submit all tasks with ThreadPoolExecutor
    with futures.ThreadPoolExecutor(max_workers=FILL_WORKERS) as executor:
        future_map = {
            executor.submit(process_question, qid): qid
            for qid in sorted(empty_ids, key=int)
        }
        
        for fut in futures.as_completed(future_map):
            qid = future_map[fut]
            try:
                qid, answer, error = fut.result()
            except Exception as exc:
                qid = future_map[fut]
                answer, error = "", exc
            
            with results_lock:
                completed += 1
                if error:
                    print(f"[{completed}/{total}] id={qid} [FAIL] {type(error).__name__}: {error}")
                    still_failed.append((qid, str(error)))
                elif answer:
                    print(f"[{completed}/{total}] id={qid} [OK] {answer[:50]}..." if len(answer) > 50 else f"[{completed}/{total}] id={qid} [OK] {answer}")
                    success_count += 1
                else:
                    print(f"[{completed}/{total}] id={qid} [WARN] empty result")
                    still_failed.append((qid, "empty result"))
                
                # Update answer in list
                if qid in answer_index:
                    answers[answer_index[qid]]["answer"] = answer
                else:
                    answers.append({"id": int(qid), "answer": answer})
                    answer_index[qid] = len(answers) - 1
    
    # Sort answers by ID before writing
    answers.sort(key=lambda x: int(x["id"]))
    
    # Write updated answers
    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        for row in answers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Append to error log
    if still_failed:
        with ERRORS_LOG.open("a", encoding="utf-8") as f:
            for qid, err in still_failed:
                f.write(f"id={qid} error={err}\n")
    
    print(f"\n=== Fill Empty Summary ===")
    print(f"Total empty: {len(empty_ids)}")
    print(f"Success: {success_count}")
    print(f"Still empty/failed: {len(still_failed)}")
    if still_failed:
        print(f"Failed IDs: {[qid for qid, _ in still_failed]}")


if __name__ == "__main__":
    main()
