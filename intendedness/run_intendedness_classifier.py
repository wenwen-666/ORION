#!/usr/bin/env python3
"""Classify test case behavior changes as regression or clean using an LLM."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LLM_MODEL, OPENAI_COMPATIBLE_BASE_URL, build_openai_client_kwargs, resolve_llm_api_key
from openai import OpenAI

from prompts.base_prompt import QUESTION_BLOCK, SYSTEM_PROMPT


PROMPT_VERSION = "default"
PROMPT_INPUT_MODE = "default"


def normalize_prompt_version(prompt_version: str) -> str:
    normalized = (prompt_version or PROMPT_VERSION).strip()
    if normalized != PROMPT_VERSION:
        raise ValueError(f"Only prompt version '{PROMPT_VERSION}' is supported.")
    return normalized


def get_user_prompt_mode(prompt_version: str) -> str:
    normalize_prompt_version(prompt_version)
    return PROMPT_INPUT_MODE


def get_execution_time_ms(run_data: Dict[str, Any]) -> Optional[float]:
    if run_data is None:
        return None
    if run_data.get("execution_time_ms") is not None:
        try:
            return float(run_data["execution_time_ms"])
        except (TypeError, ValueError):
            return None
    if run_data.get("execution_time") is not None:
        try:
            return float(run_data["execution_time"]) * 1000.0
        except (TypeError, ValueError):
            return None
    return None


def get_memory_mb(run_data: Dict[str, Any]) -> Optional[float]:
    if run_data is None:
        return None
    value = run_data.get("memory_mb")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_question_block(prompt_version: str) -> str:
    normalize_prompt_version(prompt_version)
    return QUESTION_BLOCK


def call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    api_key = resolve_llm_api_key()
    # Configure this endpoint if you use a different OpenAI-compatible API provider.
    base_url = OPENAI_COMPATIBLE_BASE_URL
    client = OpenAI(**build_openai_client_kwargs(api_key=api_key, base_url=base_url))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def parse_llm_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse LLM response as JSON: {e}", file=sys.stderr)
        return {
            "Q1": "unknown",
            "Q2": "unknown",
            "Q3": "unknown",
            "Q4": "unknown",
            "Q5": "unknown",
            "regression_label": "unknown",
            "confidence": 0.2,
            "reason": f"Failed to parse LLM response: {e}",
            "internal_error": True,
            "classification_kind": "behavior",
        }

    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        if q not in result:
            result[q] = "unknown"
        elif result[q] not in {"yes", "no", "unknown"}:
            result[q] = "unknown"

    llm_label = result.get("regression_label", "")
    if llm_label not in {"regression", "clean", "unknown"}:
        q_answers = [result[q] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]]
        computed_label = "regression" if q_answers[0] == "yes" and any(a == "yes" for a in q_answers[1:]) else "clean"
        result["regression_label"] = computed_label

    conf = result.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    result["confidence"] = max(0.0, min(1.0, conf))
    result.setdefault("reason", "")
    result.setdefault("classification_kind", "behavior")
    return result


def build_user_prompt(input_data: dict, case: dict, prompt_version: str = PROMPT_VERSION) -> str:
    prompt_version = normalize_prompt_version(prompt_version)

    pr_number = input_data.get("pr_number")
    repo = input_data.get("repo")
    pr_title = input_data.get("pr_title", "")
    pr_url = input_data.get("pr_url", "")
    pr_description = input_data.get("pr_description", "")
    pr_metadata = input_data.get("pr_metadata", {}) or {}
    pr_intent = input_data.get("pr_intent", {}) or {}

    case_id = case.get("case_id")
    entry_type = case.get("entry_type")
    behavior = case.get("behavior", {}) or {}
    before = behavior.get("before", {}) or {}
    after = behavior.get("after", {}) or {}
    diff_summary = behavior.get("diff_summary", "")
    diff_types = behavior.get("diff_types", []) or []

    exec_time_before = get_execution_time_ms(before)
    exec_time_after = get_execution_time_ms(after)
    memory_before = get_memory_mb(before)
    memory_after = get_memory_mb(after)

    lines = []
    lines.append(f"PR Info: {pr_title} (PR #{pr_number})")
    lines.append(f"Repo: {repo}")
    lines.append(f"URL: {pr_url}")
    lines.append("")

    if pr_description:
        lines.append("PR Description:")
        if len(pr_description) > 500:
            lines.append(pr_description[:500] + "... [truncated]")
        else:
            lines.append(pr_description)
        lines.append("")

    changed_files = pr_metadata.get("changed_files", []) or []
    if changed_files:
        lines.append(f"Changed files: {len(changed_files)} files")
        for path in changed_files[:10]:
            lines.append(f"  - {path}")
        if len(changed_files) > 10:
            lines.append(f"  ... and {len(changed_files) - 10} more")
        lines.append("")

    lines.append(f"Pre-commit: {pr_metadata.get('pre_commit', '')}")
    lines.append(f"Post-commit: {pr_metadata.get('post_commit', '')}")
    lines.append("")

    lines.append("=== PR INTENT HINTS ===")
    lines.append(f"Change type: {pr_intent.get('change_type', 'unknown')}")
    lines.append(f"Expected behavior change: {pr_intent.get('expected_behavior_change', 'unknown')}")
    expectations = pr_intent.get("high_level_expectations", []) or []
    if expectations:
        lines.append("High-level expectations:")
        for item in expectations[:10]:
            lines.append(f"  - {item}")
        if len(expectations) > 10:
            lines.append(f"  ... and {len(expectations) - 10} more")
    else:
        lines.append("High-level expectations: none")
    lines.append("")

    lines.append(f"Case ID: {case_id}")
    lines.append(f"Entry type: {entry_type}")
    lines.append(f"Oracle region kind: {case.get('oracle_region_kind', 'unknown')}")
    lines.append("")

    lines.append("=== OBJECTIVE EXECUTION DATA ===")
    lines.append(f"Behavior summary: {diff_summary}")
    lines.append(f"Diff types: {', '.join(diff_types)}")
    lines.append("")

    if exec_time_before is not None and exec_time_after is not None:
        lines.append("## PERFORMANCE METRICS:")
        lines.append(f"  Execution time BEFORE: {exec_time_before:.2f} ms")
        lines.append(f"  Execution time AFTER:  {exec_time_after:.2f} ms")
        if exec_time_after > 0 and exec_time_before > 0:
            ratio = exec_time_after / exec_time_before
            change_pct = (ratio - 1) * 100
            lines.append(f"  Ratio (after/before): {ratio:.2f}x")
            lines.append(f"  Change: {change_pct:+.1f}%")
            if ratio >= 2.0:
                lines.append("  WARNING: >=2x slowdown detected")
        lines.append("")

    if memory_before is not None and memory_after is not None:
        lines.append("## MEMORY USAGE:")
        lines.append(f"  Memory BEFORE: {memory_before:.2f} MB")
        lines.append(f"  Memory AFTER:  {memory_after:.2f} MB")
        if memory_after > 0 and memory_before > 0:
            ratio = memory_after / memory_before
            change_pct = (ratio - 1) * 100
            lines.append(f"  Ratio: {ratio:.2f}x")
            lines.append(f"  Change: {change_pct:+.1f}%")
        lines.append("")

    lines.append("BEFORE Execution:")
    lines.append(f"  - status: {before.get('status')}")
    lines.append(f"  - exception_type: {before.get('exception_type', 'None')}")
    lines.append(f"  - exception_message: {before.get('exception_message', 'None')}")
    lines.append(f"  - return_value_repr: {before.get('return_value_repr', 'None')}")
    stdout_before = before.get("stdout", "")
    if stdout_before:
        if len(stdout_before) > 300:
            stdout_before = stdout_before[:300] + "..."
        lines.append(f"  - stdout: {stdout_before}")
    lines.append("")

    lines.append("AFTER Execution:")
    lines.append(f"  - status: {after.get('status')}")
    lines.append(f"  - exception_type: {after.get('exception_type', 'None')}")
    lines.append(f"  - exception_message: {after.get('exception_message', 'None')}")
    lines.append(f"  - return_value_repr: {after.get('return_value_repr', 'None')}")
    stdout_after = after.get("stdout", "")
    if stdout_after:
        if len(stdout_after) > 300:
            stdout_after = stdout_after[:300] + "..."
        lines.append(f"  - stdout: {stdout_after}")
    lines.append("")

    lines.extend(get_question_block(prompt_version).splitlines())
    lines.append("")
    return "\n".join(lines)


def classify_case(
    case: dict,
    input_data: dict,
    model: str,
    system_prompt: str,
    prompt_version: str,
    user_prompt: Optional[str] = None,
) -> dict:
    behavior = case.get("behavior", {}) or {}
    before = behavior.get("before") or {}
    after = behavior.get("after") or {}
    diff_types = behavior.get("diff_types", []) or []

    exec_before = get_execution_time_ms(before)
    exec_after = get_execution_time_ms(after)
    if exec_before is not None and exec_after is not None and exec_before > 0:
        ratio = exec_after / exec_before
        if ratio >= 2.0:
            return {
                "Q1": "yes",
                "Q2": "no",
                "Q3": "no",
                "Q4": "no",
                "Q5": "yes",
                "regression_label": "regression",
                "confidence": 0.9,
                "reason": f"Severe performance regression: {ratio:.1f}x slowdown (from {exec_before:.2f}ms to {exec_after:.2f}ms).",
                "classification_kind": "performance_regression",
            }

    if "infra_error" in diff_types:
        return {
            "Q1": "unknown",
            "Q2": "unknown",
            "Q3": "unknown",
            "Q4": "unknown",
            "Q5": "unknown",
            "regression_label": "unknown",
            "confidence": 0.2,
            "reason": "Infrastructure error; cannot determine regression based on execution data.",
            "classification_kind": "infra_error",
        }

    if "new_function" in diff_types:
        return {
            "Q1": "yes",
            "Q2": "no",
            "Q3": "no",
            "Q4": "no",
            "Q5": "no",
            "regression_label": "clean",
            "confidence": 0.8,
            "reason": "New function added; no regression (based on execution data).",
            "classification_kind": "new_function",
        }

    if not behavior.get("has_behavior_diff", False) or behavior.get("diff_summary") == "no behavior change":
        return {
            "Q1": "no",
            "Q2": "no",
            "Q3": "no",
            "Q4": "no",
            "Q5": "no",
            "regression_label": "clean",
            "confidence": 0.9,
            "reason": "No behavior difference detected; not a regression.",
            "classification_kind": "behavior",
        }

    if user_prompt is None:
        user_prompt = build_user_prompt(input_data, case, prompt_version)

    try:
        llm_output = call_llm(system_prompt, user_prompt, model)
        classification = parse_llm_response(llm_output)
        classification["classification_kind"] = "behavior"
    except Exception as e:
        classification = {
            "Q1": "unknown",
            "Q2": "unknown",
            "Q3": "unknown",
            "Q4": "unknown",
            "Q5": "unknown",
            "regression_label": "unknown",
            "confidence": 0.2,
            "reason": f"LLM call failed: {e} (fallback to unknown)",
            "classification_kind": "behavior",
        }

    return classification


def summarize_pr_regression(cases: list) -> dict:
    total_cases = len(cases)
    regression_cases = []
    unknown_cases = []
    low_confidence_cases = []

    for case in cases:
        cls = case.get("classification", {})
        reg_label = cls.get("regression_label", "unknown")
        conf = cls.get("confidence", 0.0)
        if reg_label == "regression":
            regression_cases.append(case)
        if reg_label == "unknown":
            unknown_cases.append(case)
        if conf < 0.7:
            low_confidence_cases.append(case)

    pr_label = "regression" if regression_cases else "clean"
    needs_review = bool(regression_cases or unknown_cases or low_confidence_cases)
    return {
        "regression_label": pr_label,
        "needs_human_review": needs_review,
        "summary": {
            "total_cases": total_cases,
            "regression_cases": len(regression_cases),
            "unknown_cases": len(unknown_cases),
            "low_confidence_cases": len(low_confidence_cases),
        },
    }


def build_prompt_dump_record(case: dict, user_prompt: str) -> dict:
    return {
        "case_id": case.get("case_id"),
        "entry_type": case.get("entry_type"),
        "oracle_region_kind": case.get("oracle_region_kind"),
        "target_functions": case.get("target_functions", []),
        "user_prompt": user_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify test case behavior changes using LLM")
    parser.add_argument("--pr-number", type=int, required=True, help="PR number")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument("--base-dir", type=str, default=os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1])), help="Base directory")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model to use")
    parser.add_argument("--prompt-version", type=str, default=PROMPT_VERSION, help="Prompt configuration to use. Only 'default' is supported.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix to add to output filename")
    args = parser.parse_args()

    pr_number = args.pr_number
    repo = args.repo
    base_dir = args.base_dir
    model = args.model
    try:
        prompt_version = normalize_prompt_version(args.prompt_version)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    prompt_mode = get_user_prompt_mode(prompt_version)
    system_prompt = SYSTEM_PROMPT
    print(f"Using prompt version: {prompt_version}")

    if args.input_dir:
        input_path = os.path.join(args.input_dir, repo, f"pr_{pr_number}_regression_input.json")
    else:
        input_path = os.path.join(base_dir, "regression", "output", repo, f"pr_{pr_number}_regression_input.json")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading regression input from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    cases = input_data.get("cases", [])
    print(f"Processing {len(cases)} test cases with model {model}...")

    prompt_dump = {
        "pr_number": pr_number,
        "repo": repo,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_input_mode": prompt_mode,
        "system_prompt": system_prompt,
        "question_block": get_question_block(prompt_version),
        "cases": [],
    }

    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id")
        print(f"[{i}/{len(cases)}] Classifying case: {case_id}")
        user_prompt = build_user_prompt(input_data, case, prompt_version)
        prompt_dump["cases"].append(build_prompt_dump_record(case, user_prompt))
        classification = classify_case(
            case=case,
            input_data=input_data,
            model=model,
            system_prompt=system_prompt,
            prompt_version=prompt_version,
            user_prompt=user_prompt,
        )
        case["classification"] = classification
        print(f"  -> {classification['regression_label']} (confidence: {classification['confidence']:.2f})")

    pr_level_classification = summarize_pr_regression(cases)
    total_cases = len(cases)
    behavior_cases = [c for c in cases if c["classification"].get("classification_kind") == "behavior"]
    infra_error_cases = [c for c in cases if c["classification"].get("classification_kind") == "infra_error"]
    new_function_cases = [c for c in cases if c["classification"].get("classification_kind") == "new_function"]
    behavior_clean = sum(1 for c in behavior_cases if c["classification"]["regression_label"] == "clean")
    behavior_regression = sum(1 for c in behavior_cases if c["classification"]["regression_label"] == "regression")
    clean = sum(1 for c in cases if c["classification"]["regression_label"] == "clean")
    regression = sum(1 for c in cases if c["classification"]["regression_label"] == "regression")

    summary = {
        "total_cases": total_cases,
        "clean": clean,
        "regression": regression,
        "behavior_cases": len(behavior_cases),
        "behavior_clean": behavior_clean,
        "behavior_regression": behavior_regression,
        "infra_error_cases": len(infra_error_cases),
        "new_function_cases": len(new_function_cases),
    }

    regression_cases = []
    clean_cases = []
    for case in cases:
        cls = case.get("classification", {})
        reg_label = cls.get("regression_label")
        case_detail = {
            "case_id": case.get("case_id"),
            "from_spec": case.get("from_spec"),
            "entry_type": case.get("entry_type"),
            "oracle_region_kind": case.get("oracle_region_kind"),
            "oracle_region_tag": case.get("oracle_region_tag"),
            "target_functions": case.get("target_functions", []),
            "behavior": case.get("behavior", {}),
            "classification": cls,
        }
        probe_source = case.get("probe_source_code")
        if probe_source:
            case_detail["probe_source_code"] = probe_source
        if reg_label == "regression":
            regression_cases.append(case_detail)
        else:
            clean_cases.append(case_detail)

    output = {
        "pr_number": pr_number,
        "repo": repo,
        "pr_title": input_data.get("pr_title", ""),
        "pr_url": input_data.get("pr_url", ""),
        "prompt_version": prompt_version,
        "prompt_input_mode": prompt_mode,
        "model": model,
        "summary": summary,
        "pr_level_classification": pr_level_classification,
        "regression_cases": regression_cases,
        "clean_cases": clean_cases,
        "cases": regression_cases + clean_cases,
    }

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, "regression", "output", repo)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"pr_{pr_number}_regression_output{args.output_suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    prompt_dump_path = os.path.join(output_dir, f"pr_{pr_number}_regression_prompt_dump{args.output_suffix}.json")
    with open(prompt_dump_path, "w", encoding="utf-8") as f:
        json.dump(prompt_dump, f, indent=2, ensure_ascii=False)

    print(f"\n[PR {pr_number}] regression classification completed:")
    print(f"  - total cases: {total_cases}")
    print(f"  - clean: {clean}")
    print(f"  - regression: {regression}")
    print(f"  - behavior cases: {len(behavior_cases)} (clean: {behavior_clean}, regression: {behavior_regression})")
    print(f"  - new_function cases: {len(new_function_cases)}")
    print(f"  - infra_error cases: {len(infra_error_cases)}")
    print(f"  - PR label: {pr_level_classification['regression_label']}")
    print(f"  - needs_human_review: {pr_level_classification['needs_human_review']}")
    print(f"\n  - output: {output_path}")
    print(f"  - prompt dump: {prompt_dump_path}")


if __name__ == "__main__":
    main()
