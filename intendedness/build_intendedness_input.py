#!/usr/bin/env python3
"""Assemble the Stage 7 classifier input for a single PR.

This script merges PR metadata, behavior-diff output, oracle data, and PR
intent into the structured JSON consumed by the intendedness classifier.
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Constants
MAX_DIFF_CHARS = 16000
MAX_SNIPPET_LINES = 80


def load_json(path: str, required: bool = False):
    """Load JSON file, return None if not found (unless required)."""
    if not os.path.exists(path):
        if required:
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"WARNING: Optional file not found: {path}", file=sys.stderr)
            return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        if required:
            print(f"ERROR: Failed to load required file {path}: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"WARNING: Failed to load optional file {path}: {e}", file=sys.stderr)
            return None


def extract_diff_snippet(diff_lines: list, function_id: str, max_lines: int = MAX_SNIPPET_LINES) -> str:
    """
    Extract relevant diff lines for a given function_id.
    Uses the last component of function_id (e.g., 'get_fixed_timezone' from 'marshmallow.utils.get_fixed_timezone').
    """
    short_name = function_id.split(".")[-1]
    relevant_lines = []

    for line in diff_lines:
        if short_name in line:
            relevant_lines.append(line)
            if len(relevant_lines) >= max_lines:
                break

    return "\n".join(relevant_lines)


def build_function_snippets(diff_lines: list, func_map: dict) -> dict:
    """Build a map of function_id -> diff snippet."""
    snippets = {}
    for function_id in func_map.keys():
        snippet = extract_diff_snippet(diff_lines, function_id)
        snippets[function_id] = snippet
    return snippets


def build_case_entry(case: dict, func_map: dict, function_snippets: dict) -> dict:
    """Build a single case entry for the intendedness input."""
    # Basic metadata
    case_id = case.get("case_id")
    from_spec = case.get("from_spec")
    entry_type = case.get("entry_type")
    oracle_region_kind = case.get("oracle_region_kind")
    oracle_region_tag = case.get("oracle_region_tag")
    target_functions = case.get("target_functions", [])

    # Extract probe source code if available
    probe_source_code = case.get("probe_code")

    # Behavior object
    behavior_obj = {
        "has_behavior_diff": case.get("has_behavior_diff", False),
        "diff_types": case.get("diff_types", []),
        "diff_summary": case.get("diff_summary", ""),
        "before": case.get("before", {}),
        "after": case.get("after", {})
    }

    # Build relevant oracles
    relevant_oracles = []
    for func_id in target_functions:
        fo = func_map.get(func_id)
        if not fo:
            continue

        relevant_oracles.append({
            "function_id": func_id,
            "code_diff_summary": fo.get("code_diff_summary", []),
            "must_preserve_region": fo.get("must_preserve_region", []),
            "may_change_region": fo.get("may_change_region", []),
            "behavior_notes": fo.get("behavior_notes", []),
            "control_flow_hints": fo.get("control_flow_hints", {}),
            "diff_snippet": function_snippets.get(func_id, "")
        })

    # Assemble case entry
    case_entry = {
        "case_id": case_id,
        "from_spec": from_spec,
        "entry_type": entry_type,
        "oracle_region_kind": oracle_region_kind,
        "oracle_region_tag": oracle_region_tag,
        "target_functions": target_functions,
        "behavior": behavior_obj,
        "relevant_oracles": relevant_oracles
    }

    # Add probe source code if available
    if probe_source_code:
        case_entry["probe_source_code"] = probe_source_code

    return case_entry


def main():
    parser = argparse.ArgumentParser(
        description="Build intendedness input JSON from PR metadata and behavior diffs"
    )
    parser.add_argument("--pr-number", type=int, required=True, help="PR number")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1])),
        help="Base directory"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Custom PR metadata directory (default: {base-dir}/pr_metadata)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: {base-dir}/regression/output)"
    )

    args = parser.parse_args()

    pr_number = args.pr_number
    repo = args.repo
    base_dir = args.base_dir

    # Use custom metadata directory if provided, otherwise use default
    if args.metadata_dir:
        metadata_base = args.metadata_dir
    else:
        metadata_base = os.path.join(base_dir, "pr_metadata")

    # Resolve input paths (support both pr_<number>.json and <number>.json formats)
    pr_metadata_path_1 = os.path.join(
        metadata_base, repo, f"pr_{pr_number}.json"
    )
    pr_metadata_path_2 = os.path.join(
        metadata_base, repo, f"{pr_number}.json"
    )

    if os.path.exists(pr_metadata_path_1):
        pr_metadata_path = pr_metadata_path_1
    elif os.path.exists(pr_metadata_path_2):
        pr_metadata_path = pr_metadata_path_2
    else:
        pr_metadata_path = pr_metadata_path_1  # Will fail with clear error message
    behavior_path = os.path.join(
        base_dir, "test_executor", "output", repo, f"pr_{pr_number}_behavior.json"
    )
    function_oracle_path = os.path.join(
        base_dir, "function_oracle_generation", "output", repo, f"pr_{pr_number}_function_oracles.json"
    )
    intent_path = os.path.join(
        base_dir, "intent_extraction", "output", repo, f"pr_{pr_number}_intent.json"
    )

    # Load required files
    print(f"Loading PR metadata from {pr_metadata_path}...")
    pr_metadata = load_json(pr_metadata_path, required=True)

    print(f"Loading behavior data from {behavior_path}...")
    behavior_data = load_json(behavior_path, required=True)

    # Load optional files
    print(f"Loading function oracles from {function_oracle_path}...")
    function_oracle_data = load_json(function_oracle_path, required=False)

    print(f"Loading PR intent from {intent_path}...")
    intent_data = load_json(intent_path, required=False)

    # Extract PR metadata fields
    pr_title = pr_metadata.get("title", "")
    pr_url = pr_metadata.get("url", "")
    pr_description = pr_metadata.get("description", "")
    pre_commit = pr_metadata.get("pre_commit", "")
    post_commit = pr_metadata.get("post_commit", "")
    changed_files = pr_metadata.get("changed_files", [])
    changed_functions = pr_metadata.get("changed_functions", [])
    code_diff_raw = pr_metadata.get("diff", "")

    # Truncate diff if too large
    code_diff_truncated = False
    if len(code_diff_raw) > MAX_DIFF_CHARS:
        code_diff = code_diff_raw[:MAX_DIFF_CHARS]
        code_diff_truncated = True
        print(f"WARNING: Code diff truncated from {len(code_diff_raw)} to {MAX_DIFF_CHARS} chars")
    else:
        code_diff = code_diff_raw

    diff_lines = code_diff.splitlines()

    # Extract PR intent (if available)
    if intent_data:
        intent_obj = intent_data.get("intent", {}) or {}
        change_type = intent_obj.get("change_type", "unknown")
        expected_behavior_change = intent_obj.get("expected_behavior_change", "unknown")
        high_level_expectations = intent_obj.get("high_level_expectations", [])
    else:
        change_type = "unknown"
        expected_behavior_change = "unknown"
        high_level_expectations = []

    # Build function oracle map
    func_map = {}
    if function_oracle_data:
        oracles = (
            function_oracle_data.get("function_oracles") or
            function_oracle_data.get("functions") or
            []
        )
        func_map = {fo["function_id"]: fo for fo in oracles if "function_id" in fo}
        print(f"Loaded {len(func_map)} function oracles")
    else:
        print("No function oracles available")

    # Build function diff snippets
    function_snippets = build_function_snippets(diff_lines, func_map)

    # Build case entries
    cases_in = behavior_data.get("cases", [])
    cases_out = []

    print(f"Processing {len(cases_in)} test cases...")
    for case in cases_in:
        case_entry = build_case_entry(case, func_map, function_snippets)
        cases_out.append(case_entry)

    # Compute stats
    total_cases = len(cases_out)
    cases_with_diff = sum(
        1 for c in cases_out if c["behavior"].get("has_behavior_diff")
    )
    cases_with_oracle = sum(
        1 for c in cases_out if c.get("relevant_oracles")
    )

    # Build final output
    output = {
        "pr_number": pr_number,
        "repo": repo,
        "pr_title": pr_title,
        "pr_url": pr_url,
        "pr_description": pr_description,
        "pr_metadata": {
            "changed_files": changed_files,
            "changed_functions": changed_functions,
            "pre_commit": pre_commit,
            "post_commit": post_commit
        },
        "pr_intent": {
            "change_type": change_type,
            "expected_behavior_change": expected_behavior_change,
            "high_level_expectations": high_level_expectations
        },
        "code_diff": code_diff,
        "code_diff_truncated": code_diff_truncated,
        "cases": cases_out,
        "stats": {
            "total_cases": total_cases,
            "cases_with_diff": cases_with_diff,
            "cases_with_oracle": cases_with_oracle
        }
    }

    # Write output
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, repo)
    else:
        output_dir = os.path.join(base_dir, "regression", "output", repo)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"pr_{pr_number}_regression_input.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n[PR {pr_number}] regression input built:")
    print(f"  - total cases: {total_cases}")
    print(f"  - cases with diff: {cases_with_diff}")
    print(f"  - cases with oracle info: {cases_with_oracle}")
    print(f"  - code diff truncated: {code_diff_truncated}")
    print(f"  - output: {output_path}")


if __name__ == "__main__":
    main()
