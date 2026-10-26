#!/usr/bin/env python3
"""Organize intendedness outputs into the final result layout.

This stage copies the classifier outputs into the repository's categorized
result folders so downstream inspection can use a stable directory structure.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _find_existing_input(output_base: Path, pr_number: int) -> Optional[Path]:
    candidates = [
        output_base / f"pr_{pr_number}_regression_input.json",
        output_base / f"pr_{pr_number}_intendedness_input.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def _find_existing_outputs(output_base: Path, pr_number: int) -> List[Path]:
    patterns = [
        f"pr_{pr_number}_regression_output.json",
        f"pr_{pr_number}_intendedness_output.json",
    ]
    outputs: List[Path] = []
    for pattern in patterns:
        outputs.extend(sorted(output_base.glob(pattern)))
    seen = set()
    unique_outputs = []
    for path in outputs:
        if path.name not in seen:
            seen.add(path.name)
            unique_outputs.append(path)
    return unique_outputs


def _find_prompt_dumps(output_base: Path, pr_number: int) -> List[Path]:
    return sorted(output_base.glob(f"pr_{pr_number}_regression_prompt_dump.json"))


def _select_primary_output(outputs: List[Path]) -> Optional[Path]:
    return outputs[0] if outputs else None


def _load_classification(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _category_from_pr_label(data: Dict) -> str:
    pr_cls = data.get("pr_level_classification", {}) or {}
    pr_label = pr_cls.get("regression_label", pr_cls.get("label", "unknown"))
    return "intended" if pr_label in {"clean", "intended"} else "unintended"


def organize_pr_results(repo: str, pr_number: int, base_dir: str, output_dir: str = None):
    output_base = Path(output_dir) if output_dir else Path(base_dir) / "intendedness" / "output" / repo
    input_file = _find_existing_input(output_base, pr_number)
    output_files = _find_existing_outputs(output_base, pr_number)
    prompt_dump_files = _find_prompt_dumps(output_base, pr_number)

    if not output_files:
        print(f"WARNING: Output files not found for PR {pr_number}, skipping")
        return None

    primary_output = _select_primary_output(output_files)
    data = _load_classification(primary_output)
    category = _category_from_pr_label(data)
    pr_cls = data.get("pr_level_classification", {}) or {}

    all_results_dir = output_base / "all_results"
    intended_dir = output_base / "intended"
    unintended_dir = output_base / "unintended"
    for directory in (all_results_dir, intended_dir, unintended_dir):
        directory.mkdir(parents=True, exist_ok=True)

    category_dir = intended_dir if category == "intended" else unintended_dir

    if input_file and input_file.exists():
        target_name = f"pr_{pr_number}_intendedness_input.json"
        shutil.copy2(input_file, all_results_dir / target_name)
        shutil.copy2(input_file, category_dir / target_name)

    for path in output_files + prompt_dump_files:
        shutil.copy2(path, all_results_dir / path.name)
        shutil.copy2(path, category_dir / path.name)

    return {
        "pr_number": pr_number,
        "category": category,
        "needs_review": pr_cls.get("needs_human_review", True),
        "primary_output": primary_output.name,
        "available_outputs": [path.name for path in output_files],
        "total_cases": data.get("summary", {}).get("total_cases", 0),
        "clean_cases": data.get("summary", {}).get("clean", data.get("summary", {}).get("intended", 0)),
        "regression_cases": data.get("summary", {}).get("regression", data.get("summary", {}).get("unintended", 0)),
    }


def create_summary_file(repo: str, base_dir: str, results: list, output_dir: str = None):
    output_base = Path(output_dir) if output_dir else Path(base_dir) / "intendedness" / "output" / repo
    intended_prs = [r for r in results if r["category"] == "intended"]
    unintended_prs = [r for r in results if r["category"] == "unintended"]

    summary = {
        "repo": repo,
        "total_prs": len(results),
        "intended_prs": len(intended_prs),
        "unintended_prs": len(unintended_prs),
        "intended_pr_list": [r["pr_number"] for r in intended_prs],
        "unintended_pr_list": [r["pr_number"] for r in unintended_prs],
        "details": results,
    }

    summary_path = output_base / "all_results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme_files(output_base, summary)
    return summary


def write_readme_files(output_base: Path, summary: dict):
    main_lines = [
        f"# Intendedness Results - {summary['repo']}",
        "",
        f"Total PRs: {summary['total_prs']}",
        f"Intended/Clean PRs: {summary['intended_prs']}",
        f"Unintended/Regression PRs: {summary['unintended_prs']}",
        "",
        "The final categorization is based on the intendedness output.",
        "The classifier output and prompt dump are preserved in both `all_results/` and the category folder.",
    ]
    (output_base / "README.md").write_text("\n".join(main_lines) + "\n", encoding="utf-8")
    (output_base / "intended" / "README.md").write_text(
        "Clean PRs live here. The classifier output and prompt dump are copied here.\n",
        encoding="utf-8",
    )
    (output_base / "unintended" / "README.md").write_text(
        "Regression PRs live here. The classifier output and prompt dump are copied here.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Organize intendedness results by category")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1])),
        help="Base directory",
    )
    parser.add_argument(
        "--pr-numbers",
        type=str,
        default=None,
        help="Comma-separated list of PR numbers. If omitted, organize every PR with outputs in the directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory containing classifier results",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else Path(args.base_dir) / "intendedness" / "output" / args.repo
    if args.pr_numbers:
        pr_numbers = [int(item) for item in args.pr_numbers.split(",") if item.strip()]
    else:
        pr_numbers = sorted(
            {
                int(path.stem.split("_")[1])
                for path in _find_existing_outputs(output_base, 0)  # dummy call for pattern shape not used
            }
        )
        if not pr_numbers:
            found = set()
            for path in output_base.glob("pr_*_regression_output*.json"):
                found.add(int(path.stem.split("_")[1]))
            for path in output_base.glob("pr_*_intendedness_output*.json"):
                found.add(int(path.stem.split("_")[1]))
            pr_numbers = sorted(found)

    results = []
    for pr_number in pr_numbers:
        result = organize_pr_results(args.repo, pr_number, args.base_dir, args.output_dir)
        if result:
            results.append(result)

    if results:
        summary = create_summary_file(args.repo, args.base_dir, results, args.output_dir)
        print(f"Organized {len(results)} PRs for {args.repo}")
        print(f"Summary: {output_base / 'all_results' / 'summary.json'}")
    else:
        print("No results organized")


if __name__ == "__main__":
    main()
