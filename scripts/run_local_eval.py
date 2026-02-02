from __future__ import annotations

import concurrent.futures as futures
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import solve  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_PATH = DATA_DIR / "question.jsonl"
OUTPUT_PATH = DATA_DIR / "answer.jsonl"
EVAL_WORKERS = int(os.getenv("EVAL_WORKERS", "8"))


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

    with futures.ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        future_map = {
            executor.submit(solve, row["question"]): idx
            for idx, row in enumerate(questions)
        }
        for fut in futures.as_completed(future_map):
            idx = future_map[fut]
            row = questions[idx]
            try:
                answer = fut.result()
            except Exception as exc:
                answer = ""
                with error_log.open("a", encoding="utf-8") as logf:
                    logf.write(f"id={row['id']} error={exc!r}\n")
            results[idx] = {"id": row["id"], "answer": answer}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
