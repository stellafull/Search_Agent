from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import solve  # noqa: E402

DATA_DIR = ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "question.jsonl"
OUTPUT_PATH = DATA_DIR / "answer_one.jsonl"


def load_question(path: Path, question_id: Optional[str] = None) -> dict:
    """Load a question by ID, or the first question if no ID specified."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if question_id is None:
                return row  # Return first question
            if str(row.get("id")) == str(question_id):
                return row
    if question_id:
        raise RuntimeError(f"Question with id={question_id} not found in {path}")
    raise RuntimeError("question.jsonl is empty")


def list_questions(path: Path, limit: int = 20) -> None:
    """List available questions with their IDs."""
    print(f"Available questions (showing first {limit}):\n")
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id", "?")
            question = row.get("question", "")[:80]
            print(f"  [{qid}] {question}...")
            count += 1
            if count >= limit:
                print(f"\n  ... use --list-all to see all questions")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on a single question")
    parser.add_argument(
        "question_id",
        nargs="?",
        default=None,
        help="ID of the question to test (default: first question)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available question IDs",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all question IDs",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: data/answer_one.jsonl)",
    )
    args = parser.parse_args()

    if args.list or args.list_all:
        limit = 9999 if args.list_all else 20
        list_questions(QUESTIONS_PATH, limit=limit)
        return

    row = load_question(QUESTIONS_PATH, args.question_id)
    print(f"Testing question [{row['id']}]:")
    print(f"  {row['question']}\n")

    answer = solve(row["question"])
    result = {"id": row["id"], "answer": answer}

    output_path = Path(args.output) if args.output else OUTPUT_PATH
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False))
        f.write("\n")

    print(f"\n{'='*60}")
    print(f"Question: {row['question']}")
    print(f"Answer:   {answer}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
