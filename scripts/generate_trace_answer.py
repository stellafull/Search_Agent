#!/usr/bin/env python3
"""
Match traces with questions and generate trace_answer.jsonl.

Reads traces.jsonl outputs, matches question field with question.jsonl,
and generates answer file with question id and final_answer.
"""

import json
from pathlib import Path


def generate_trace_answers(
    traces_file: str = "data/traces.jsonl",
    questions_file: str = "data/question.jsonl",
    output_file: str = "data/trace_answer.jsonl",
):
    """
    Match traces with questions and generate answer file.

    Args:
        traces_file: Path to traces.jsonl
        questions_file: Path to question.jsonl
        output_file: Output path for trace_answer.jsonl
    """
    # Load questions and build question -> id mapping
    print(f"📖 Loading questions from {questions_file}")
    question_to_id: dict[str, int] = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            question_to_id[data["question"]] = data["id"]

    print(f"✅ Loaded {len(question_to_id)} questions")

    # Load traces and match with questions
    print(f"📖 Loading traces from {traces_file}")
    answers: dict[int, list[str]] = {}  # id -> list of answers (collect all for duplicates)
    matched_count = 0
    unmatched_count = 0

    with open(traces_file, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line.strip())
            outputs = trace.get("outputs", {})
            question = outputs.get("question")
            final_answer = outputs.get("final_answer")

            if not question or not final_answer:
                print(f"⚠️  Trace {trace.get('id', 'unknown')}: missing question or final_answer")
                continue

            if question in question_to_id:
                qid = question_to_id[question]
                if qid not in answers:
                    answers[qid] = []
                answers[qid].append(final_answer)
                matched_count += 1
            else:
                unmatched_count += 1
                print(f"⚠️  No match for question: {question[:50]}...")

    print(f"✅ Matched {matched_count} traces, {unmatched_count} unmatched")
    print(f"📊 Unique questions answered: {len(answers)}")
    
    # Count duplicates
    duplicates = {qid: ans for qid, ans in answers.items() if len(ans) > 1}
    if duplicates:
        print(f"📋 Questions with multiple answers: {len(duplicates)}")
        for qid, ans_list in duplicates.items():
            print(f"   id={qid}: {len(ans_list)} answers")

    # Write output sorted by id (include all questions, empty answer if no trace)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all question ids
    all_ids = sorted(question_to_id.values())
    
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in all_ids:
            if qid in answers:
                # Join multiple answers with comma, deduplicate while preserving order
                ans_list = answers[qid]
                seen = set()
                unique_answers = []
                for ans in ans_list:
                    if ans not in seen:
                        seen.add(ans)
                        unique_answers.append(ans)
                answer_str = ", ".join(unique_answers)
            else:
                answer_str = ""
            data = {"id": qid, "answer": answer_str}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    missing_count = len(all_ids) - len(answers)
    if missing_count > 0:
        print(f"⚠️  {missing_count} questions have no answer (output as empty string)")

    print(f"💾 Saved {len(answers)} answers to {output_path}")

    return answers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate trace_answer.jsonl from traces")
    parser.add_argument(
        "--traces",
        "-t",
        type=str,
        default="data/traces.jsonl",
        help="Path to traces.jsonl",
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=str,
        default="data/question.jsonl",
        help="Path to question.jsonl",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/trace_answer.jsonl",
        help="Output path for trace_answer.jsonl",
    )

    args = parser.parse_args()

    generate_trace_answers(
        traces_file=args.traces,
        questions_file=args.questions,
        output_file=args.output,
    )
