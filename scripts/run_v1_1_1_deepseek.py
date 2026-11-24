#!/usr/bin/env python3
"""Evaluate DeepSeek V4 Flash on the published CORE-LLM-Bench v1.1.1 dataset."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from itertools import islice, zip_longest
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from evaluate_v1_1_final_offline import (
    SAGEQA_SOURCE_COMMIT,
    answer_set_scores,
    gold_for_representation,
    parse_response,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output/deepseek-v4-flash-v1.1.1"
DATASET_SHA256 = "0a051b7ccc021a0fafa1852ee3dd02d49ea79631d67adb86e33b360a7fccb803"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://gcapi.cn/v1"
RETURNED_MODELS = {MODEL, "deepseek-flash"}
REPRESENTATIONS = {
    "AR": ("Abstracted", "#3078ba"),
    "NL": ("Natural language", "#e37444"),
    "FS": ("Formal syntax", "#1ca47c"),
}
TASK_TOTALS = {"BQA": 6032, "OEQA": 3016}
PARAMETERS = {
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 0.9,
    "thinking": {"type": "disabled"},
}


def load_questions(path: Path, *, with_prompts: bool = True) -> list[dict]:
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != DATASET_SHA256:
            raise ValueError("Parquet SHA-256 differs from the published v1.1.1 dataset.")
    columns = [
        "task_id", "task_group", "gold_answer", "ar_gold_answer", "primitive_reasoning_tags",
    ]
    if with_prompts:
        columns += [
            "answer_type", "nl_question", "nl_context", "fs_query", "fs_context",
            "ar_question", "ar_context",
        ]
    parquet = pq.ParquetFile(path)
    questions = [
        row
        for batch in parquet.iter_batches(batch_size=256, columns=columns)
        for row in batch.to_pylist()
    ]
    if (
        Counter(row["task_group"] for row in questions) != TASK_TOTALS
        or len({row["task_id"] for row in questions}) != len(questions)
    ):
        raise ValueError("Unexpected benchmark membership.")
    for row in questions:
        tags = row["primitive_reasoning_tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)
        row["tag_count"] = len(set(tags) - {"M"})
        if not row["tag_count"]:
            raise ValueError(f"No primitive reasoning tags for task {row['task_id']}.")
    return questions


def planned_observations(questions: list[dict], completed: set[tuple]) -> list[tuple]:
    shuffled = sorted(questions, key=lambda row: row["task_id"])
    random.Random(0).shuffle(shuffled)
    strata = defaultdict(list)
    for row in shuffled:
        strata[row["task_group"], row["tag_count"]].append(row)
    return [
        (row, rep)
        for round_rows in zip_longest(*strata.values())
        for row in round_rows if row is not None
        for rep in REPRESENTATIONS
        if (row["task_id"], rep) not in completed
    ]


def initialize(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS run_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            config TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS responses (
            task_id INTEGER NOT NULL,
            representation TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            returned_model TEXT NOT NULL,
            finish_reason TEXT,
            usage TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            PRIMARY KEY (task_id, representation)
        );
    """)
    config = json.dumps({
        "dataset_sha256": DATASET_SHA256,
        "request_model": MODEL,
        "endpoint": BASE_URL,
        "parameters": PARAMETERS,
        "prompt_renderer": "phase7d_finalize_experiment.prompt_for",
    }, sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO run_config VALUES (1, ?)", (config,))
        if db.execute("SELECT config FROM run_config WHERE id = 1").fetchone()[0] != config:
            raise ValueError("The response database belongs to a different run configuration.")


async def collect(
    db: sqlite3.Connection, pending: list[tuple], completed: int, total: int, workers: int
) -> int:
    from phase7d_finalize_experiment import prompt_for

    api_key = os.environ.get("CORE_LLM_BENCH_API_KEY")
    if not api_key:
        raise ValueError("Set CORE_LLM_BENCH_API_KEY for API execution.")
    queue = iter(pending)
    failures = Counter()
    started = time.monotonic()
    last_progress = started
    cooldown_until = started
    concurrency = min(workers, len(pending))

    async def run_worker(client: httpx.AsyncClient) -> None:
        nonlocal completed, last_progress, cooldown_until
        for row, representation in queue:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt_for(row, representation)}],
                **PARAMETERS,
            }
            for attempt in range(1, 6):
                while (delay := cooldown_until - time.monotonic()) > 0:
                    await asyncio.sleep(delay)
                failure = None
                try:
                    response = await client.post("chat/completions", json=payload)
                except httpx.RequestError as error:
                    failure = type(error).__name__
                else:
                    try:
                        data = response.json()
                    except ValueError:
                        data = {}
                    if not isinstance(data, dict):
                        data = {}
                    if response.is_success and "choices" in data:
                        choices = data["choices"]
                        if choices and isinstance(choices[0], dict):
                            choice = choices[0]
                            message = choice.get("message") or {}
                            content = message.get("content")
                            returned_model = data.get("model")
                            if returned_model not in RETURNED_MODELS:
                                raise RuntimeError("The API returned an unrecognized model identifier.")
                            if isinstance(content, str) and content.strip():
                                with db:
                                    db.execute(
                                        "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                                        (
                                            row["task_id"], representation,
                                            content.replace(api_key, "[redacted]"), returned_model,
                                            choice.get("finish_reason"), json.dumps(data.get("usage") or {}),
                                            attempt,
                                        ),
                                    )
                                completed += 1
                                if completed % 250 == 0 or time.monotonic() - last_progress >= 20:
                                    last_progress = time.monotonic()
                                    print(f"{completed:,}/{total:,} responses saved", flush=True)
                                break
                        failure = "empty_or_invalid_completion"
                    else:
                        error = data.get("error") or {}
                        code = str(error.get("code", "")).lower() if isinstance(error, dict) else ""
                        status = response.status_code
                        failure = f"HTTP {status}"
                        if status in {401, 402, 403, 404} or "quota" in code or "balance" in code:
                            raise RuntimeError(f"The endpoint rejected the run ({failure}).")
                        if status == 429:
                            retry_after = response.headers.get("Retry-After", "")
                            delay = float(retry_after) if retry_after.isdigit() else 3.0 * attempt
                            cooldown_until = max(cooldown_until, time.monotonic() + delay)
                        elif status not in {408, 409, 425} and status < 500 and not response.is_success:
                            failures[failure] += 1
                            break
                if attempt == 5:
                    failures[failure] += 1
                else:
                    await asyncio.sleep(2 ** (attempt - 1) + random.random())

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=BASE_URL, headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(90, connect=15), limits=limits, follow_redirects=False,
    ) as client:
        tasks = [asyncio.create_task(run_worker(client)) for _ in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    print(
        f"{completed:,}/{total:,} responses saved in this run and previous runs "
        f"({time.monotonic() - started:.1f}s this session). Failures: {dict(failures)}",
        flush=True,
    )
    return completed


def summarize(questions: list[dict], db: sqlite3.Connection) -> tuple[list[dict], dict]:
    by_id = {row["task_id"]: row for row in questions}
    group_sizes = Counter((row["task_group"], row["tag_count"]) for row in questions)
    scores = defaultdict(list)
    parsing = Counter()
    finish = Counter()
    models = Counter()
    tokens = Counter()
    for task_id, rep, content, model, finish_reason, raw_usage, attempts in db.execute(
        "SELECT * FROM responses ORDER BY task_id, representation"
    ):
        row = by_id[task_id]
        task = row["task_group"]
        parsed = parse_response(content, task)
        em, f1, _, _ = answer_set_scores(parsed["answer"], gold_for_representation(row, rep))
        scores[task, rep, row["tag_count"]].append(em if task == "BQA" else f1)
        parsing[parsed["status"]] += 1
        finish[finish_reason or "unspecified"] += 1
        models[model] += 1
        usage = json.loads(raw_usage)
        for key in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens"):
            tokens[key] += usage.get(key) or 0
        tokens["technical_retries"] += attempts - 1

    points = []
    for (task, count), expected in sorted(group_sizes.items()):
        for rep in REPRESENTATIONS:
            values = scores[task, rep, count]
            points.append({
                "task": task,
                "representation": rep,
                "reasoning_tag_count": count,
                "metric": "Answer Exact Match" if task == "BQA" else "Answer F1",
                "score_percent": 100 * math.fsum(values) / len(values) if values else None,
                "n_scored": len(values),
                "n_expected": expected,
            })
    expected = len(questions) * len(REPRESENTATIONS)
    summary = {
        "dataset": "CORE-LLM-Bench v1.1.1",
        "dataset_sha256": DATASET_SHA256,
        "requested_model": MODEL,
        "endpoint": BASE_URL,
        "request_parameters": PARAMETERS,
        "returned_models": dict(models),
        "expected_observations": expected,
        "scored_observations": models.total(),
        "complete": models.total() == expected,
        "sageqa_evaluator_source_commit": SAGEQA_SOURCE_COMMIT,
        "tag_count": "Number of distinct primitive_reasoning_tags in the release, excluding M",
        "plot_tag_count_range": {"BQA": [1, 4], "OEQA": [1, 6]},
        "parser_status_counts": dict(parsing),
        "finish_reason_counts": dict(finish),
        "usage": dict(tokens),
    }
    return points, summary


def write_report(output: Path, points: list[dict], summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = "accuracy_by_reasoning_tag_count"
    if not summary["complete"]:
        stem += "_partial"
    with (output / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (output / f"{stem}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.21, top=0.76, wspace=0.26)
    fig.suptitle("Accuracy by reasoning-tag count, split by representation", y=0.96)
    caption = f"{MODEL}  |  CORE-LLM-Bench v1.1.1"
    if not summary["complete"]:
        caption += f"  |  Partial run ({summary['scored_observations']:,}/{summary['expected_observations']:,})"
    fig.text(0.5, 0.86, caption, ha="center", color="#50565b", fontsize=10)
    for ax, task, metric in zip(axes, TASK_TOTALS, ("Exact-match accuracy (%)", "Answer F1 (%)")):
        low, high = summary["plot_tag_count_range"][task]
        task_points = [
            point for point in points
            if point["task"] == task and low <= point["reasoning_tag_count"] <= high
        ]
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("Number of reasoning tags")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted({point["reasoning_tag_count"] for point in task_points}))
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)
        ax.spines[["right", "top"]].set_visible(False)
        for rep, (label, color) in REPRESENTATIONS.items():
            series = [point for point in task_points if point["representation"] == rep]
            ax.plot(
                [point["reasoning_tag_count"] for point in series],
                [point["score_percent"] for point in series],
                label=f"{rep} ({label})", color=color, marker="o", markersize=4,
                linewidth=1.6, clip_on=False,
            )
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.text(0.075, 0.045, "The CSV also reports OEQA at seven tags (n = 2).",
             color="#50565b", fontsize=8)
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"Report written to {output / stem} (PNG, CSV, JSON)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=OUTPUT / "benchmark.parquet")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--limit", type=int, help="Evaluate at most this many new observations.")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Label and plot incomplete coverage.")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive.")

    questions = load_questions(args.benchmark, with_prompts=not args.plot_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(args.output_dir / "responses.sqlite3")) as db:
        initialize(db)
        completed = set(db.execute("SELECT task_id, representation FROM responses"))
        pending = planned_observations(questions, completed)
        total = len(questions) * len(REPRESENTATIONS)
        if len(completed) + len(pending) != total:
            raise ValueError("The response database contains unknown observations.")
        if not args.plot_only and pending:
            batch = list(islice(pending, args.limit))
            print(f"Starting {len(batch):,} requests with {len(completed):,}/{total:,} already saved", flush=True)
            count = asyncio.run(collect(db, batch, len(completed), total, args.workers))
            if count < total and args.limit is not None and not args.allow_partial:
                return
        points, summary = summarize(questions, db)
        if not summary["scored_observations"] or not (summary["complete"] or args.allow_partial):
            raise SystemExit("Incomplete coverage. Resume API execution or use --allow-partial to plot saved responses.")
        write_report(args.output_dir, points, summary)


if __name__ == "__main__":
    main()
