#!/usr/bin/env python3
"""Summarize pre/post probe execution results into behavior-diff JSON.

This stage normalizes noisy runtime details, compares the pre-commit and
post-commit probe outcomes, and writes the behavior summary consumed by the
intendedness classifier.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ADDRESS_PATTERN = re.compile(r" at 0x[0-9a-fA-F]+")
# Matches conda env paths like /home/.../conda_envs/pkg_pr1234_pre/... or /home/.../conda_envs/pkg_pr1234_post/...
CONDA_ENV_PATTERN = re.compile(r"/[^\s\"']*conda_envs/[^/\s\"']+_(?:pre|post)/")
# Matches /tmp/tmpXXXXXX style temp paths
TMP_PATH_PATTERN = re.compile(r"/tmp/tmp[A-Za-z0-9_]+")


def clean_repr(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = ADDRESS_PATTERN.sub(" at 0x...", str(text))
    text = CONDA_ENV_PATTERN.sub("/<conda_env>/", text)
    text = TMP_PATH_PATTERN.sub("/tmp/<tmpfile>", text)
    return text


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        print(f"[Warning] Failed to load {path}: {exc}")
    return None


def extract_behavior_snapshot(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robustly extract exception info from either observations or top-level error.
    """
    observations = result_json.get("observations", {}) or {}
    exc_type = observations.get("exception_type")
    exc_msg = observations.get("exception_message")

    if not exc_type and result_json.get("status") == "exception":
        raw_error = result_json.get("error", "")
        if raw_error and ": " in raw_error:
            exc_type, exc_msg = raw_error.split(": ", 1)
        else:
            exc_type = exc_type or "RuntimeError"
            exc_msg = exc_msg or raw_error

    return {
        "status": result_json.get("status"),
        "exception_type": exc_type,
        "exception_message": clean_repr(exc_msg),
        "return_value_repr": clean_repr(observations.get("return_value")),
        "stdout": clean_repr(result_json.get("stdout", "")),
        "stderr": clean_repr(result_json.get("stderr", "")),
        "error_kind": result_json.get("error_kind"),
        "execution_time": result_json.get("execution_time"),
    }


def compute_diff(before: Dict[str, Any], after: Dict[str, Any]) -> (List[str], str, bool):
    diff_types: List[str] = []
    before_error = before.get("error_kind")
    after_error = after.get("error_kind")
    before_status = before.get("status")
    after_status = after.get("status")

    # Regression via removed symbols: used-to-work, now import error
    if before_status == "success" and after_error == "import_error":
        diff_types.append("removed_feature")
        return diff_types, "Regression: Feature/Symbol removed (ImportError)", True

    # Syntax errors are always infra issues regardless of side
    if before_error == "syntax_error" or after_error == "syntax_error":
        diff_types.append("infra_error")
        return diff_types, "Invalid Probe (Syntax Error)", False

    if before_error in ("import_error", "syntax_error") and after_error in ("import_error", "syntax_error"):
        diff_types.append("infra_error")
        return diff_types, "Invalid probe (Import/Syntax error in both versions)", False

    # system_exit in both versions means the probe itself failed (e.g. bad import path), treat as infra error
    before_sys_exit = before.get("error_kind") == "system_exit"
    after_sys_exit = after.get("error_kind") == "system_exit"
    if before_sys_exit and after_sys_exit:
        diff_types.append("infra_error")
        return diff_types, "Invalid probe (SystemExit in both versions)", False

    if (before_error in ("import_error", "syntax_error")) and after_status == "success":
        diff_types.append("new_function")
    elif before_error == "call_error" and after_status == "success":
        diff_types.append("api_change_fixed")
    elif before_status == "success" and after_error == "call_error":
        diff_types.append("api_breakage")
    else:
        if before_status != after_status:
            diff_types.append("status_changed")
        if before.get("exception_type") != after.get("exception_type"):
            diff_types.append("exception_type_changed")
        if before.get("exception_message") != after.get("exception_message"):
            diff_types.append("exception_message_changed")
        if before.get("return_value_repr") != after.get("return_value_repr"):
            diff_types.append("return_value_changed")
        if before.get("stdout") != after.get("stdout"):
            diff_types.append("stdout_changed")
        if before.get("stderr") != after.get("stderr"):
            diff_types.append("stderr_changed")

        # Check for performance regression (>2x slowdown)
        before_time = before.get("execution_time")
        after_time = after.get("execution_time")
        if before_time and after_time and before_time > 0.1:  # Only check if baseline > 100ms
            if after_time > before_time * 2.0:
                diff_types.append("performance_regression")
            elif after_time < before_time * 0.5:
                diff_types.append("performance_improvement")

    summary_parts: List[str] = []
    if "new_function" in diff_types:
        summary_parts.append("New function (ImportError -> OK)")
    elif "removed_feature" in diff_types:
        summary_parts.append("Regression: symbol removed (OK -> ImportError)")
    elif "api_breakage" in diff_types:
        summary_parts.append("API breakage (OK -> CallError)")
    else:
        if "status_changed" in diff_types:
            summary_parts.append(f"status: {before_status}->{after_status}")
        if "exception_type_changed" in diff_types:
            summary_parts.append(f"exc: {before.get('exception_type')}->{after.get('exception_type')}")
        if "return_value_changed" in diff_types:
            summary_parts.append(f"return value changed: {before.get('return_value_repr')!r} -> {after.get('return_value_repr')!r}")
        if "stdout_changed" in diff_types:
            summary_parts.append("stdout changed")
        if "stderr_changed" in diff_types:
            summary_parts.append("stderr changed")
        if "exception_message_changed" in diff_types:
            summary_parts.append(f"exc message changed: {before.get('exception_message')!r} -> {after.get('exception_message')!r}")
        if "performance_regression" in diff_types:
            before_time = before.get("execution_time", 0)
            after_time = after.get("execution_time", 0)
            summary_parts.append(f"perf regression: {before_time:.3f}s -> {after_time:.3f}s ({after_time/before_time:.1f}x)")
        if "performance_improvement" in diff_types:
            before_time = before.get("execution_time", 0)
            after_time = after.get("execution_time", 0)
            summary_parts.append(f"perf improvement: {before_time:.3f}s -> {after_time:.3f}s ({before_time/after_time:.1f}x faster)")

    diff_summary = "; ".join(summary_parts) if summary_parts else "no behavior change"
    has_behavior_diff = len(diff_types) > 0 and "infra_error" not in diff_types
    return diff_types, diff_summary, has_behavior_diff


def convert_probe_to_case(probe: Dict[str, Any], idx: int) -> Dict[str, Any]:
    case_id = probe.get("probe_id") or probe.get("test_id") or f"probe_{idx}"
    targets = probe.get("target_functions") or []
    if not targets and probe.get("target_function_id"):
        targets = [probe["target_function_id"]]
    return {
        "case_id": str(case_id),
        "from_spec": probe.get("source", "probe"),
        "entry_type": probe.get("type", "unit"),
        "oracle_region_kind": probe.get("oracle_region_kind"),
        "target_functions": targets,
        "probe_code": probe.get("probe_code") or probe.get("code"),
    }


def load_cases_for_pr(base_dir: Path, repo: str, pr_number: int) -> (List[Dict[str, Any]], Optional[str]):
    probe_path = base_dir / "test_probe_generation" / "output" / repo / f"pr_{pr_number}_test_specs.json"
    data = load_json_file(probe_path)
    if not data:
        return [], None
    probes = data.get("probes") or data.get("specs") or []
    cases = [convert_probe_to_case(p, idx) for idx, p in enumerate(probes) if isinstance(p, dict)]
    return cases, str(probe_path)


def clean_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only objective facts from a case for prompt optimization."""
    return {
        "case_id": case.get("case_id"),
        "target_functions": case.get("target_functions", []),
        "probe_code": case.get("probe_code"),
        "before": case.get("before"),
        "after": case.get("after"),
        "has_behavior_diff": case.get("has_behavior_diff", False),
    }


def process_case(case: Dict[str, Any], pre_dir: Path, post_dir: Path) -> Dict[str, Any]:
    case_id = case["case_id"]
    pre_json = load_json_file(pre_dir / f"{case_id}.json")
    post_json = load_json_file(post_dir / f"{case_id}.json")

    result = {
        "case_id": case_id,
        "from_spec": case.get("from_spec"),
        "entry_type": case.get("entry_type"),
        "oracle_region_kind": case.get("oracle_region_kind"),
        "target_functions": case.get("target_functions", []),
        "probe_code": case.get("probe_code"),
    }

    if pre_json is None or post_json is None:
        result.update({
            "before": None,
            "after": None,
            "has_behavior_diff": False,
            "diff_types": ["infra_error"],
            "diff_summary": "Missing output file",
        })
        return result

    before = extract_behavior_snapshot(pre_json)
    after = extract_behavior_snapshot(post_json)
    diff_types, diff_summary, has_diff = compute_diff(before, after)

    result.update({
        "before": before,
        "after": after,
        "has_behavior_diff": has_diff,
        "diff_types": diff_types,
        "diff_summary": diff_summary,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize behavior differences")
    parser.add_argument("--pr", "--pr-number", dest="pr", type=int, required=True, help="PR number")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--base-dir", default=os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1])))
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    cases, source_path = load_cases_for_pr(base_dir, args.repo, args.pr)
    if not cases:
        print(f"[Error] No probes found for PR {args.pr}")
        return 1

    pre_dir = base_dir / "test_outputs" / args.repo / f"pr_{args.pr}" / "pre"
    post_dir = base_dir / "test_outputs" / args.repo / f"pr_{args.pr}" / "post"

    results = [process_case(case, pre_dir, post_dir) for case in cases]

    stats = {
        "total_cases": len(results),
        "infra_errors": sum(1 for r in results if "infra_error" in r["diff_types"]),
        "behavior_diffs": sum(1 for r in results if r["has_behavior_diff"]),
    }

    output_dir = base_dir / "test_executor" / "output" / args.repo
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pr_{args.pr}_behavior.json"
    output_path.write_text(json.dumps({"cases": results, "stats": stats, "probe_source": source_path}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Behavior summary written to {output_path}")
    print(f"Stats: {stats}")

    # Generate clean version for prompt optimization
    clean_cases = [clean_case(case) for case in results]
    # Filter out cases with missing probe_code or execution results
    valid_clean_cases = [
        c for c in clean_cases
        if c["probe_code"] and c["before"] and c["after"]
    ]
    clean_output_path = output_dir / f"pr_{args.pr}_behavior_clean.json"
    clean_output_path.write_text(
        json.dumps({"cases": valid_clean_cases, "total_cases": len(valid_clean_cases)}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Clean behavior written to {clean_output_path}")
    print(f"  Valid cases: {len(valid_clean_cases)}/{len(results)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
