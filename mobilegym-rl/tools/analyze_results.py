#!/usr/bin/env python3
"""Unified MobileGym rollout statistics.

Input is one ``results.jsonl`` (or a directory containing such files).  The
script writes normalized records, grouped metrics, per-task rollout metrics,
summary JSON, and optional PNG charts.  It deliberately has no project
imports, so it can be run from the host or from the training container.

``results.jsonl`` produced by older MobileGym runs does not persist reward,
wrong-answer, or token lengths.  In that case reward is reconstructed with
the same rules as ``cookbooks/mobilegym/evaluator.py`` and lengths are read
from the sibling ``trajectory`` directory using a regex token estimate.  A
Transformers tokenizer may be supplied for exact token counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUPS = ("suite", "difficulty", "scope", "composition")
SIGNALS = (
    "progress",
    "goal_success",
    "success",
    "passed",
    "false_complete",
    "overdue",
    "wrong_answer_submitted",
)
TERMINATIONS = ("COMPLETE", "ABORT", "MAX_STEPS", "FORMAT_ERROR", "REPETITIVE_LOOP", "ERROR", "UNKNOWN")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def quantile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else None


def variance(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.pvariance(vals) if vals else None


def regex_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def prompt_text(obj: Any) -> str:
    """Extract textual content from arbitrary chat/prompt JSON."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n".join(prompt_text(x) for x in obj)
    if isinstance(obj, dict):
        # Avoid counting image URLs/base64 and metadata as prompt tokens.
        preferred = [obj[k] for k in ("text", "content", "prompt", "input") if k in obj]
        if preferred:
            return "\n".join(prompt_text(x) for x in preferred)
        return "\n".join(prompt_text(v) for k, v in obj.items() if k not in {"image", "images", "url"})
    return ""


def extract_checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Use persisted goal_checks, or judge.issues in older result files."""
    checks = row.get("goal_checks")
    if isinstance(checks, list):
        return checks
    judge = row.get("judge") or {}
    issues = judge.get("issues")
    return issues if isinstance(issues, list) else []


def wrong_answer(goal_checks: Any) -> bool:
    checks = goal_checks if isinstance(goal_checks, list) else []
    submitted = next((c for c in checks if str(c.get("field", "")) == "answer_sheet.submitted"), None)
    content = [
        c for c in checks
        if str(c.get("field", "")).startswith("answer_sheet.")
        and str(c.get("field", "")) != "answer_sheet.submitted"
    ]
    return bool(submitted is not None and submitted.get("actual") is True and content and any(not as_bool(c.get("passed")) for c in content))


def progress_without_submitted(goal_checks: Any, default: float) -> float:
    checks = [c for c in (goal_checks if isinstance(goal_checks, list) else []) if str(c.get("field", "")) != "answer_sheet.submitted"]
    return sum(as_bool(c.get("passed")) for c in checks) / len(checks) if checks else default


def derive_reward(row: dict[str, Any], wrong: bool) -> float:
    execution = row.get("execution") or {}
    judge = row.get("judge") or {}
    progress = as_float(row.get("progress", judge.get("progress", 0.0)))
    goal_success = as_bool(judge.get("success", row.get("goal_success", row.get("success", False))))
    success = as_bool(row.get("is_success", row.get("success", False)))
    clean = as_bool(judge.get("clean", row.get("clean", True)))
    finished = as_bool(execution.get("finished", row.get("finished", False)))
    stop_reason = str(execution.get("stop_reason", row.get("stop_reason", "")) or "")
    has_error = as_bool(row.get("is_error", False)) or bool(execution.get("error"))
    # Only a full ``goal_checks`` list can support exact removal of the
    # submitted bonus.  Older results persist failed checks only in
    # ``judge.issues``; keep their recorded progress in that case rather than
    # treating the incomplete issue list as the complete denominator.
    full_checks = row.get("goal_checks")
    reward = progress_without_submitted(full_checks, progress) if wrong and isinstance(full_checks, list) else progress
    if goal_success and not clean:
        reward *= 0.8
    if stop_reason == "ABORT" and goal_success and not success and reward > 0:
        reward *= 0.5
    if as_bool(row.get("overdue_termination", row.get("overdue", False))):
        reward *= 0.5
    if finished and not goal_success and not has_error and reward > 0:
        reward *= 0.8
    return reward


def trajectory_lengths(row: dict[str, Any], results_path: Path, tokenizer: Any = None) -> tuple[int | None, int | None, str]:
    """Return (response_tokens, prompt_tokens, source), if trajectory exists."""
    for field in ("response_length", "completion_length"):
        if field in row:
            response = as_int(row[field], -1)
            prompt = as_int(row.get("prompt_length", row.get("input_length", -1)), -1)
            if response >= 0 or prompt >= 0:
                return (response if response >= 0 else None, prompt if prompt >= 0 else None, "result_fields")

    root = results_path.parent / "trajectory"
    if not root.is_dir():
        return None, None, "missing"
    trial = row.get("trial_id")
    suffix = f"_t{trial}" if trial is not None else ""
    task = str(row.get("id", row.get("task_name", ""))).replace(".", "_").replace("/", "_")
    candidates = [p for p in root.glob(f"*{suffix}") if p.is_dir()] if suffix else []
    candidates.sort(key=lambda p: (0 if p.name.startswith(task) else 1, p.name))
    if not candidates:
        return None, None, "missing"
    traj = candidates[0] / "trajectory.json"
    if not traj.exists():
        return None, None, "missing"
    try:
        steps = json.loads(traj.read_text(encoding="utf-8"))
    except Exception:
        return None, None, "unreadable"
    if not isinstance(steps, list):
        return None, None, "unreadable"
    responses, prompts = [], []
    for step in steps:
        if not isinstance(step, dict):
            continue
        d = candidates[0]
        rp = d / str(step.get("model_response_path", ""))
        pp = d / str(step.get("model_prompt_path", ""))
        if rp.exists():
            txt = rp.read_text(encoding="utf-8", errors="ignore")
            responses.append(len(tokenizer.encode(txt, add_special_tokens=False)) if tokenizer else regex_tokens(txt))
        if pp.exists():
            try:
                obj = json.loads(pp.read_text(encoding="utf-8"))
                txt = prompt_text(obj)
            except Exception:
                txt = pp.read_text(encoding="utf-8", errors="ignore")
            prompts.append(len(tokenizer.encode(txt, add_special_tokens=False)) if tokenizer else regex_tokens(txt))
    return (sum(responses) if responses else None, sum(prompts) if prompts else None, "trajectory_tokenizer" if tokenizer else "trajectory_regex")


def normalize(row: dict[str, Any], results_path: Path, tokenizer: Any = None) -> dict[str, Any]:
    execution, judge = row.get("execution") or {}, row.get("judge") or {}
    raw_stop = str(execution.get("stop_reason", row.get("stop_reason", "")) or "UNKNOWN").upper()
    stop = raw_stop
    if stop not in TERMINATIONS:
        if "LOOP" in stop or "REPETITIVE" in stop:
            stop = "REPETITIVE_LOOP"
        elif "FORMAT" in stop:
            stop = "FORMAT_ERROR"
        elif "MAX" in stop or "TRUNCAT" in stop:
            stop = "MAX_STEPS"
        else:
            stop = "UNKNOWN"
    checks = extract_checks(row)
    wrong = wrong_answer(checks) or as_bool(row.get("wrong_answer_submitted", False))
    response_len, prompt_len, length_source = trajectory_lengths(row, results_path, tokenizer)
    record = {
        "id": row.get("id", row.get("task_name", "")),
        "task_name": row.get("task_name", row.get("id", "")),
        "suite": row.get("suite", "unknown"),
        "difficulty": row.get("difficulty", "unknown"),
        "scope": row.get("scope", "unknown"),
        "composition": row.get("composition", "unknown"),
        "trial_id": row.get("trial_id", ""),
        "steps": as_int(execution.get("steps", row.get("steps", 0))),
        "stop_reason_raw": raw_stop,
        "stop_reason": stop,
        "progress": as_float(row.get("progress", judge.get("progress", 0.0))),
        "raw_progress": as_float(row.get("raw_progress", row.get("progress", judge.get("progress", 0.0)))),
        "goal_success": int(as_bool(judge.get("success", row.get("goal_success", row.get("success", False))))),
        "success": int(as_bool(row.get("is_success", row.get("success", False)))),
        "passed": int(as_bool(judge.get("passed", row.get("passed", False)))),
        "false_complete": int(as_bool(row.get("false_complete", False))),
        "overdue": int(as_bool(row.get("overdue_termination", row.get("overdue", False)))),
        "wrong_answer_submitted": int(wrong),
        "repeated_action_count": as_int(execution.get("repeated_action_count", row.get("repeated_action_count", 0))),
        "no_state_change_count": as_int(execution.get("no_state_change_count", row.get("no_state_change_count", 0))),
        "clean": int(as_bool(judge.get("clean", row.get("clean", True)))),
        "response_length": response_len,
        "prompt_length": prompt_len,
        "length_source": length_source,
    }
    explicit_reward = row.get("reward")
    record["reward"] = as_float(explicit_reward) if explicit_reward is not None else derive_reward(row, wrong)
    if row.get("reward_breakdown"):
        record["reward_source"] = "reward_sidecar"
        breakdown = row.get("reward_breakdown") or {}
        record["reward_base"] = as_float(breakdown.get("base_reward"), record["raw_progress"])
        applied = breakdown.get("penalties") or {}
        for name in ("clean", "abort", "overdue", "premature", "format_error", "repetitive_loop", "max_steps", "infra_error", "repeated_action", "no_state_change"):
            record[f"penalty_{name}"] = applied.get(name)
    else:
        record["reward_source"] = "result_field" if explicit_reward is not None else ("derived_evaluator" if isinstance(row.get("goal_checks"), list) else "derived_evaluator_approx")
        record["reward_base"] = None
        for name in ("clean", "abort", "overdue", "premature", "format_error", "repetitive_loop", "max_steps", "infra_error", "repeated_action", "no_state_change"):
            record[f"penalty_{name}"] = None
    record["termination_class"] = row.get("termination_class", "INFRA_ERROR" if row.get("is_error") else stop)
    return record


def metric_row(group: str, value: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    out: dict[str, Any] = {"group": group, "value": value, "n": n}
    out["success_count"] = sum(r["success"] for r in records)
    out["success_rate"] = out["success_count"] / n if n else None
    for s in SIGNALS:
        vals = [r[s] for r in records]
        out[f"{s}_mean"] = mean(vals)
        if s == "progress":
            out["progress_sum"] = sum(vals)
        else:
            out[f"{s}_count"] = sum(vals)
    for name in ("steps", "response_length", "prompt_length", "reward"):
        vals = [r[name] for r in records if r[name] is not None]
        out[f"{name}_mean"] = mean(vals)
        if name == "steps":
            out["steps_p50"], out["steps_p90"], out["steps_p95"] = (quantile(vals, q) for q in (0.5, 0.9, 0.95))
        if name == "reward":
            out["reward_variance"] = variance(vals)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_reward_sidecar(results_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load evaluator-time rewards written beside ``results.jsonl``.

    The bench runner records its result before the rLLM evaluator runs, so
    exact P1 rewards live in ``reward.jsonl``.  Keying by task/trial avoids
    changing the legacy results schema or duplicating result rows.
    """
    sidecar = results_path.parent / "reward.jsonl"
    if not sidecar.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for line in sidecar.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item.get("task_id", "")), str(item.get("trial_id", "")))
            out[key] = item
    except Exception as exc:
        print(f"[warn] failed to read {sidecar}: {exc}", file=sys.stderr)
    return out


def make_charts(out: Path, records: list[dict[str, Any]], grouped: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipped charts: {exc}", file=sys.stderr)
        return False
    for group in GROUPS:
        rows = [r for r in grouped if r["group"] == group]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(5, len(rows) * 1.3), 4))
        ax.bar([str(r["value"]) for r in rows], [r["success_rate"] or 0 for r in rows])
        ax.set_ylim(0, 1); ax.set_ylabel("success rate"); ax.set_title(f"Success by {group}")
        ax.tick_params(axis="x", labelrotation=35); fig.tight_layout(); fig.savefig(out / f"success_by_{group}.png", dpi=140); plt.close(fig)
    counts = Counter(r["stop_reason"] for r in records)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(list(counts), list(counts.values())); ax.set_ylabel("rollouts"); ax.set_title("Termination reasons"); fig.tight_layout(); fig.savefig(out / "termination_counts.png", dpi=140); plt.close(fig)
    steps = [r["steps"] for r in records]
    if steps:
        fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(steps, bins=min(20, max(5, len(set(steps))))); ax.set_xlabel("interaction steps"); ax.set_ylabel("rollouts"); fig.tight_layout(); fig.savefig(out / "steps_histogram.png", dpi=140); plt.close(fig)
    if tasks:
        top = sorted(tasks, key=lambda r: r["reward_variance"] or 0, reverse=True)[:30]
        fig, ax = plt.subplots(figsize=(max(7, len(top) * .28), 5)); ax.bar(range(len(top)), [r["reward_variance"] or 0 for r in top]); ax.set_xlabel("task (sorted by reward variance)"); ax.set_ylabel("reward variance"); fig.tight_layout(); fig.savefig(out / "task_reward_variance.png", dpi=140); plt.close(fig)
    return True


def analyze(input_path: Path, output: Path, expected_rollouts: int = 8, tokenizer_name: str | None = None) -> None:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("results.jsonl"))
    if not files:
        raise SystemExit(f"No results.jsonl found under {input_path}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as exc:
            print(f"[warn] tokenizer unavailable ({exc}); using regex estimate", file=sys.stderr)
    records: list[dict[str, Any]] = []
    invalid = 0
    for file in files:
        sidecar = load_reward_sidecar(file)
        for line_no, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict): raise ValueError("JSON object expected")
                side = sidecar.get((str(row.get("id", "")), str(row.get("trial_id", ""))))
                if side:
                    # Sidecar values are authoritative because they were
                    # produced after rLLM evaluator penalties were applied.
                    row = dict(row)
                    row.update({
                        "reward": side.get("reward"),
                        "raw_progress": side.get("raw_progress", row.get("progress")),
                        "reward_breakdown": side.get("reward_breakdown"),
                        "termination_class": side.get("termination_class"),
                    })
                rec = normalize(row, file, tokenizer); rec["source_results"] = str(file); records.append(rec)
            except Exception as exc:
                invalid += 1; print(f"[warn] skipped {file}:{line_no}: {exc}", file=sys.stderr)
    write_csv(output / "records.csv", records)

    grouped: list[dict[str, Any]] = []
    for group in GROUPS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records: buckets[str(r[group])].append(r)
        grouped.extend(metric_row(group, value, buckets[value]) for value in sorted(buckets))
    write_csv(output / "grouped_metrics.csv", grouped)

    buckets = defaultdict(list)
    for r in records: buckets[str(r["id"])].append(r)
    tasks = []
    for task_id, rs in sorted(buckets.items()):
        row = {"task_id": task_id, "rollout_count": len(rs), "expected_rollouts": expected_rollouts, "complete_rollouts": int(len(rs) == expected_rollouts)}
        row.update({"success_count": sum(r["success"] for r in rs), "success_rate": mean(r["success"] for r in rs), "reward_mean": mean(r["reward"] for r in rs), "reward_variance": variance(r["reward"] for r in rs), "reward_std": math.sqrt(variance(r["reward"] for r in rs)) if variance(r["reward"] for r in rs) is not None else None, "progress_mean": mean(r["progress"] for r in rs), "steps_mean": mean(r["steps"] for r in rs), "steps_p50": quantile((r["steps"] for r in rs), .5), "steps_p90": quantile((r["steps"] for r in rs), .9), "steps_p95": quantile((r["steps"] for r in rs), .95), "false_complete_count": sum(r["false_complete"] for r in rs), "overdue_count": sum(r["overdue"] for r in rs), "wrong_answer_count": sum(r["wrong_answer_submitted"] for r in rs)})
        tasks.append(row)
    write_csv(output / "task_rollouts.csv", tasks)

    n = len(records)
    summary: dict[str, Any] = {"input_files": [str(p) for p in files], "rollouts": n, "invalid_lines": invalid, "expected_rollouts_per_task": expected_rollouts, "length_sources": dict(Counter(r["length_source"] for r in records)), "reward_sources": dict(Counter(r["reward_source"] for r in records))}
    summary["success_count"] = sum(r["success"] for r in records); summary["success_rate"] = summary["success_count"] / n if n else None
    for s in SIGNALS:
        if s == "progress":
            summary["progress_sum"] = sum(r[s] for r in records)
            continue
        summary[f"{s}_count"] = sum(r[s] for r in records); summary[f"{s}_rate"] = mean(r[s] for r in records)
    summary["termination_counts"] = dict(Counter(r["stop_reason"] for r in records))
    for name in ("steps", "response_length", "prompt_length", "reward", "progress"):
        vals = [r[name] for r in records if r[name] is not None]
        summary[f"{name}_mean"] = mean(vals); summary[f"{name}_p50"] = quantile(vals, .5); summary[f"{name}_p90"] = quantile(vals, .9); summary[f"{name}_p95"] = quantile(vals, .95)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_charts(output, records, grouped, tasks)
    print(json.dumps({"rollouts": n, "success_rate": summary["success_rate"], "output": str(output), "invalid_lines": invalid}, ensure_ascii=False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="results.jsonl or directory containing results.jsonl")
    p.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    p.add_argument("--expected-rollouts", type=int, default=8, help="expected rollouts per task (default: 8)")
    p.add_argument("--tokenizer", help="optional HF tokenizer name/path for exact prompt/response lengths")
    args = p.parse_args(); analyze(args.input, args.output, args.expected_rollouts, args.tokenizer); return 0


if __name__ == "__main__":
    raise SystemExit(main())
