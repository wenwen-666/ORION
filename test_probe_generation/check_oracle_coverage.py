#!/usr/bin/env python3
"""
Stage 4b: Oracle Coverage Checker

Analyzes test specs to determine coverage of oracle targets (must_preserve, may_change, new_paths).
Outputs a coverage report identifying which oracle items are missing test coverage.

Usage:
  python3 check_oracle_coverage.py --repo marshmallow --pr 2701 --base-dir /path/to/ORION
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

BASE_DIR_DEFAULT = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def module_from_file(file_path: str) -> str:
    """Extract module path from file path."""
    if not file_path:
        return ""
    normalized = file_path.replace("\\", "/").lstrip("./")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [p for p in normalized.split("/") if p]
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[0] == "tests":
        parts = parts[1:]
    return ".".join(parts)


def is_code_function(function_data: Dict[str, Any]) -> bool:
    """Heuristic check for code functions."""
    kind = function_data.get("kind") or function_data.get("node_kind")
    if kind:
        return kind == "code"
    return bool(function_data.get("source_after") or function_data.get("source_before"))


def is_changed_role(function_data: Dict[str, Any]) -> bool:
    """Filter out non-changed roles when present."""
    role = function_data.get("function_role") or function_data.get("role")
    if role is None:
        return True
    if role.startswith("changed"):
        return True
    return role in {"changed_code", "changed_logic", "changed_interface", "changed_behavior"}


def is_importable_entry(function: Dict[str, Any]) -> bool:
    """Check if function can be imported through import + getattr."""
    callability = function.get("callability") or {}
    is_nested = callability.get("is_nested", False)

    # Skip explicitly nested functions
    if is_nested:
        return False

    # Skip functions with <locals> in function_id (nested function marker)
    fid = function.get("function_id") or ""
    if "<locals>" in fid:
        return False

    # Skip functions with multiple dots in qualified_name that aren't class methods
    entry_kind = callability.get("entry_kind", "function")
    qualified = function.get("qualified_name", "")

    if entry_kind == "function" and qualified.count(".") > 0:
        pass

    # Check source code indentation for nested functions
    if entry_kind == "function" and not callability.get("class_qualname"):
        source = function.get("source_after") or function.get("source_before") or ""
        if source:
            for line in source.split("\n"):
                if line.strip().startswith("def "):
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:
                        return False
                    break

    return True


def build_oracle_targets(target_view: Dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract oracle targets from target_view.
    Only includes functions that can actually generate specs (aligned with Stage 4a filtering).

    Returns:
        Dict mapping function_id -> {must_preserve, may_change, new_paths}
    """
    oracle_targets: Dict[str, Dict[str, List[str]]] = {}

    for func in target_view.get("functions", []):
        oracle = func.get("oracle", {}) or {}

        # Skip functions with oracle generation errors OR intentionally skipped by budget
        if oracle.get("error") or oracle.get("skipped"):
            continue

        # Align with Stage 4a filtering: only include functions that will actually generate specs
        if not is_code_function(func):
            continue
        if not is_changed_role(func):
            continue
        if not is_importable_entry(func):
            continue
        if not module_from_file(func.get("file", "")):
            continue

        func_id = func["function_id"]
        cf = oracle.get("control_flow_hints", {}) or {}

        oracle_targets[func_id] = {
            "must_preserve": oracle.get("must_preserve_region", []) or [],
            "may_change": oracle.get("may_change_region", []) or [],
            "new_paths": cf.get("new_paths", []) or [],
        }

    return oracle_targets


def build_oracle_ref_index(specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Set[int]]]:
    """
    Build structured oracle_ref index from probes.

    Returns:
        Dict mapping function_id -> {region_name: {index1, index2, ...}}
    """
    idx: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

    for spec in specs:
        target_id = spec.get("target_function_id")
        if not target_id:
            tfs = spec.get("target_functions", [])
            if tfs:
                target_id = tfs[0]

        if not target_id:
            continue

        ref = spec.get("oracle_ref")
        if not isinstance(ref, dict):
            continue

        region = ref.get("region")
        index = ref.get("index")

        if region == "new_path":
            region = "new_paths"

        if region and isinstance(index, int):
            idx[target_id][region].add(index)

    return idx


def compute_coverage(oracle_targets, oracle_ref_index):
    """
    Compute oracle coverage for each function.

    Args:
        oracle_targets: Dict of function_id -> oracle targets
        oracle_ref_index: Dict of function_id -> {region: {indices}}

    Returns:
        List of function coverage results
    """
    functions_out = []

    for func_id, ot in oracle_targets.items():
        ref_index = oracle_ref_index.get(func_id, {})

        missing_items = []
        covered_items = []

        def check_list(kind: str, items: List[str], region_name: str):
            """Check coverage for a list of oracle items."""
            covered = 0
            for idx, item in enumerate(items):
                # Priority: structured oracle_ref match
                ref_hit = idx in ref_index.get(region_name, set())

                if ref_hit:
                    covered += 1
                    covered_items.append({
                        "kind": kind,
                        "text": item,
                        "matched_by": "oracle_ref",
                        "region": region_name,
                    })
                else:
                    missing_items.append({
                        "kind": kind,
                        "text": item,
                        "index": idx,
                        "region": region_name,
                    })
            return covered, len(items)

        # Check coverage for each oracle category
        mp_cov, mp_total = check_list("must_preserve", ot["must_preserve"], "must_preserve")
        mc_cov, mc_total = check_list("may_change", ot["may_change"], "may_change")
        np_cov, np_total = check_list("new_path", ot["new_paths"], "new_paths")

        functions_out.append({
            "function_id": func_id,
            "oracle_targets": ot,
            "coverage": {
                "must_preserve_covered": mp_cov,
                "must_preserve_total": mp_total,
                "may_change_covered": mc_cov,
                "may_change_total": mc_total,
                "new_paths_covered": np_cov,
                "new_paths_total": np_total,
            },
            "missing_items": missing_items,
            "covered_items": covered_items,
        })

    return functions_out


def main():
    ap = argparse.ArgumentParser(description="Check oracle coverage in test specs")
    ap.add_argument("--repo", required=True, help="Repository name (e.g., marshmallow)")
    ap.add_argument("--pr", type=int, required=True, help="PR number")
    ap.add_argument("--base-dir", default=BASE_DIR_DEFAULT, help="Base directory")
    args = ap.parse_args()

    base = Path(args.base_dir)
    repo = args.repo
    pr = args.pr

    # Input paths
    target_view_path = base / "target_function_view" / "output" / repo / f"pr_{pr}_targets_view.json"
    specs_path = base / "test_probe_generation" / "output" / repo / f"pr_{pr}_test_specs.json"

    # Output path
    out_path = base / "test_probe_generation" / "output" / repo / f"pr_{pr}_oracle_coverage.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate inputs exist
    if not target_view_path.exists():
        print(f"[ERROR] Target view not found: {target_view_path}")
        return 1
    if not specs_path.exists():
        print(f"[ERROR] Test specs not found: {specs_path}")
        return 1

    # Load data
    target_view = load_json(target_view_path)
    specs_data = load_json(specs_path)
    specs = specs_data.get("probes") or specs_data.get("specs") or []

    # Build oracle targets and probe index
    oracle_targets = build_oracle_targets(target_view)
    oracle_ref_index = build_oracle_ref_index(specs)

    # Compute coverage
    functions_out = compute_coverage(oracle_targets, oracle_ref_index)

    # Calculate summary statistics
    summary = {
        "total_functions": len(functions_out),
        "functions_with_missing": sum(1 for f in functions_out if f["missing_items"]),
        "total_missing_items": sum(len(f["missing_items"]) for f in functions_out),
        "total_covered_items": sum(len(f["covered_items"]) for f in functions_out),
    }

    # Build output
    out_data = {
        "pr_number": pr,
        "repo": repo,
        "functions": functions_out,
        "summary": summary,
    }

    # Write output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"[PR {pr}] Oracle coverage analysis complete")
    print(f"  Total functions: {summary['total_functions']}")
    print(f"  Functions with missing coverage: {summary['functions_with_missing']}")
    print(f"  Total missing items: {summary['total_missing_items']}")
    print(f"  Total covered items: {summary['total_covered_items']}")
    print(f"  Output: {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
