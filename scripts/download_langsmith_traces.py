#!/usr/bin/env python3
"""
Download LangSmith tracing records with filters.

Filter: Is Trace = true AND Status = success
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "src" / ".env")


def download_traces(
    project_name: str | None = None,
    output_file: str = "data/langsmith_traces.jsonl",
    limit: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
):
    """
    Download LangSmith traces with filter: is_root=True AND status=success.

    Args:
        project_name: LangSmith project name (default from env LANGSMITH_PROJECT)
        output_file: Output JSONL file path
        limit: Maximum number of traces to download (None for all)
        start_time: Filter traces after this time
        end_time: Filter traces before this time
    """
    # Initialize client
    client = Client()

    # Get project name from env if not provided
    if project_name is None:
        project_name = os.getenv("LANGSMITH_PROJECT", "tianchi_v2")

    print(f"📡 Connecting to LangSmith project: {project_name}")
    print(f"🔍 Filter: is_root=True AND status=success")

    # Build filter kwargs
    filter_kwargs = {
        "project_name": project_name,
        "is_root": True,  # Is Trace = true (root traces only)
        "error": False,  # Status = success (no errors)
    }

    if start_time:
        filter_kwargs["start_time"] = start_time
    if end_time:
        filter_kwargs["end_time"] = end_time
    if limit:
        filter_kwargs["limit"] = limit

    # Fetch traces
    print("⏳ Fetching traces...")
    traces = list(client.list_runs(**filter_kwargs))

    print(f"✅ Found {len(traces)} traces")

    # Prepare output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save traces to JSONL (outputs only)
    saved_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            trace_data = {
                "id": str(trace.id),
                "outputs": trace.outputs,
            }
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"💾 Saved {saved_count} traces to {output_path}")

    # Print summary
    if traces:
        print("\n📊 Summary:")
        total_tokens = sum(t.total_tokens or 0 for t in traces)
        total_cost = sum(t.total_cost or 0 for t in traces)
        latencies = []
        for t in traces:
            if t.latency is not None:
                lat = t.latency.total_seconds() if hasattr(t.latency, 'total_seconds') else t.latency
                latencies.append(lat)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total cost: ${total_cost:.4f}")
        print(f"   Avg latency: {avg_latency:.2f}s")

    return traces


def download_trace_with_children(
    trace_id: str,
    output_file: str = "data/trace_detail.json",
):
    """
    Download a single trace with all its child runs.

    Args:
        trace_id: The trace/run ID to download
        output_file: Output JSON file path
    """
    client = Client()

    print(f"📡 Fetching trace: {trace_id}")

    # Get the root trace
    root_run = client.read_run(trace_id)

    # Get all child runs
    child_runs = list(
        client.list_runs(
            trace_id=trace_id,
            is_root=False,
        )
    )

    print(f"✅ Found root trace + {len(child_runs)} child runs")

    def run_to_dict(run):
        return {
            "id": str(run.id),
            "name": run.name,
            "run_type": run.run_type,
            "status": run.status,
            "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "latency": run.latency.total_seconds() if run.latency else None,
            "total_tokens": run.total_tokens,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
        }

    # Build trace data
    trace_data = {
        "root": run_to_dict(root_run),
        "children": [run_to_dict(r) for r in child_runs],
    }

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved trace detail to {output_path}")

    return trace_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download LangSmith traces")
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        default=None,
        help="LangSmith project name (default: from LANGSMITH_PROJECT env)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/langsmith_traces.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of traces to download",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=None,
        help="Download traces from last N days",
    )
    parser.add_argument(
        "--trace-id",
        "-t",
        type=str,
        default=None,
        help="Download a specific trace with all children",
    )

    args = parser.parse_args()

    if args.trace_id:
        # Download specific trace with children
        output = args.output
        if output == "data/langsmith_traces.jsonl":
            output = f"data/trace_{args.trace_id[:8]}.json"
        download_trace_with_children(args.trace_id, output)
    else:
        # Download filtered traces
        start_time = None
        if args.days:
            start_time = datetime.now() - timedelta(days=args.days)

        download_traces(
            project_name=args.project,
            output_file=args.output,
            limit=args.limit,
            start_time=start_time,
        )
