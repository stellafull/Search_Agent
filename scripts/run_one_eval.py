from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import solve  # noqa: E402

DATA_DIR = ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "question.jsonl"
OUTPUT_PATH = DATA_DIR / "answer_one.jsonl"


def load_first_question(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise RuntimeError("question.jsonl is empty")


def main() -> None:
    row = load_first_question(QUESTIONS_PATH)
    answer = solve(row["question"])
    result = {"id": row["id"], "answer": answer}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False))
        f.write("\n")

    print("question:", row["question"])
    print("answer:", answer)


if __name__ == "__main__":
    main()
