#!/usr/bin/env python3
"""
Target Function View Generator (Fixed: Intersection-based Imports)

Builds per-PR target function views by integrating:
1. Impact region (changed functions + impact neighbors)
2. Function oracles (behavioral constraints)
3. PR-level intent (high-level purpose)

Output: pr_<number>_targets_view.json
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import resolve_project_checkout_from_base_dir


def _file_to_module(file_path: str) -> str:
    """Convert file path to module name with real module semantics."""
    if not file_path:
        return ""
    module = file_path.replace("\\", "/").replace("/", ".")
    if module.endswith(".py"):
        module = module[:-3]
    elif module.endswith(".pyx"):
        module = module[:-4]

    for prefix in ("src.", "lib.", "tests.", "test."):
        if module.startswith(prefix):
            module = module[len(prefix):]
    return module


# --- Behavior dimension keyword sets (heuristic, can be extended later) ---

MISSING_VALUE_TOKENS = {
    "isna", "notna", "isnull", "notnull",
    "dropna", "fillna", "nan", "null", "none",
    "missing", "nulls", "nans", "nas",
}

SIZE_TOKENS = {
    "len(", ".len(", "size", "is empty", "is_empty", "empty",
    "num_rows", "num_cols", "nrows", "ncols", "length",
    "large", "huge", "many rows", "many elements", "chunk",
}

ZERO_TOKENS = {
    "== 0", "!= 0", "< 0", "> 0", "zero", "zeros",
    "nonzero", "non-zero",
}

NEGATIVE_TOKENS = {
    "negative", "negatives", "< 0", "less than 0", "less than zero",
}

ORDERING_TOKENS = {
    "sort_values", "sort_index", "argsort", "order", "ordered", "sorted",
    "ascending", "descending", "rank", "nlargest", "nsmallest",
    "reorder", "reordered", "monotonic", "monotonic_increasing", "monotonic_decreasing",
}

BOUNDARY_TOKENS = {
    "boundary", "boundaries", "limit", "limits", "upper bound", "lower bound",
    "min_value", "max_value", "clip", "clipping", "saturate",
    "min(", "max(",
}

OVERFLOW_TOKENS = {
    "overflow", "underflow", "wraparound", "wrap-around", "overflowing",
}

MONOTONICITY_TOKENS = {
    "monotonic", "monotone", "increasing", "decreasing",
    "non-decreasing", "nondecreasing", "non-increasing", "nonincreasing",
    "non-monotonic", "nonmonotonic",
}


class TargetFunctionViewBuilder:
    """Build comprehensive target function views for test generation."""

    def __init__(
        self,
        impact_region_path: str,
        function_oracle_path: Optional[str] = None,
        pr_intent_path: Optional[str] = None,
        repo_root: Optional[str] = None,
    ):
        self.impact_region_path = Path(impact_region_path)
        self.function_oracle_path = Path(function_oracle_path) if function_oracle_path else None
        self.pr_intent_path = Path(pr_intent_path) if pr_intent_path else None

        # Load input files
        self.impact_data = self._load_json(impact_region_path)
        self.oracle_data = self._load_json(function_oracle_path) if function_oracle_path else {}
        self.intent_data = self._load_json(pr_intent_path) if pr_intent_path else {}

        self.pr_number = self.impact_data.get('pr_number')
        self.repo_root = self._resolve_repo_root(repo_root)
        self.pre_commit = self.impact_data.get('pre_commit') or self.intent_data.get('pre_commit')

    def _load_json(self, path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_repo_root(self, repo_root: Optional[str]) -> Optional[Path]:
        """Best-effort detection of the repository checkout path."""
        candidates: List[Path] = []
        if repo_root:
            candidates.append(Path(repo_root))
        repo_name = self.impact_region_path.parent.name if self.impact_region_path.parent else None
        try:
            base_root = self.impact_region_path.parents[3]
            if repo_name:
                candidates.append(resolve_project_checkout_from_base_dir(base_root, repo_name))
        except IndexError:
            pass
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None

    def _get_file_content_git(self, commit: Optional[str], file_path: Optional[str]) -> str:
        """Read historical file content via `git show` without checkout."""
        if not self.repo_root or not commit or not file_path:
            return ""
        rel_path = file_path.lstrip("./")
        try:
            result = subprocess.run(
                ["git", "show", f"{commit}:{rel_path}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout

    def _get_file_content_disk(self, file_path: Optional[str]) -> str:
        """Read the current (post-commit) version of the file from disk."""
        if not self.repo_root or not file_path:
            return ""
        candidate = self.repo_root / file_path
        if not candidate.exists() and file_path.startswith("src/"):
            candidate = self.repo_root / file_path[4:]
        if not candidate.exists():
            alt = Path(file_path)
            if alt.is_absolute() and alt.exists():
                candidate = alt
            else:
                return ""
        try:
            return candidate.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return ""

    def _extract_absolute_imports(self, source: str, module_name: str) -> Set[str]:
        """Parse source into AST and extract normalized absolute import statements."""
        if not source:
            return set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()

        imports: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    stmt = f"import {alias.name}"
                    if alias.asname:
                        stmt += f" as {alias.asname}"
                    imports.add(stmt)
            elif isinstance(node, ast.ImportFrom):
                level = getattr(node, "level", 0) or 0
                if level > 0 and module_name:
                    parts = module_name.split(".")
                    if level <= len(parts):
                        base_parts = parts[:-level]
                        if node.module:
                            base_parts.append(node.module)
                        abs_module = ".".join(base_parts)
                    else:
                        abs_module = ""
                else:
                    abs_module = node.module or ""

                if not abs_module:
                    continue

                names = ", ".join(
                    f"{alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    for alias in node.names
                )
                imports.add(f"from {abs_module} import {names}")
        return imports

    def _compute_stable_context(self, file_path: Optional[str]) -> Dict[str, Any]:
        """
        Build a stable code context by intersecting pre/post import lists.
        """
        module_name = _file_to_module(file_path) if file_path else ""
        source_post = self._get_file_content_disk(file_path)
        source_pre = self._get_file_content_git(self.pre_commit, file_path)

        imports_post = self._extract_absolute_imports(source_post, module_name)
        imports_pre = self._extract_absolute_imports(source_pre, module_name)

        if source_post and source_pre:
            stable_imports = sorted(imports_post.intersection(imports_pre))
        elif source_post:
            stable_imports = sorted(imports_post)
        else:
            stable_imports = sorted(imports_pre)

        return {
            "source": source_post,
            "imports": stable_imports,
        }

    def _extract_init_signature(self, source: str, class_qualname: Optional[str]) -> str:
        """Extract the __init__ definition text for the given class."""
        if not source or not class_qualname:
            return ""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""
        lines = source.splitlines()

        def visit(node: ast.ClassDef, stack: List[str]) -> Optional[str]:
            stack.append(node.name)
            try:
                if ".".join(stack) == class_qualname:
                    for stmt in node.body:
                        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                            start = getattr(stmt, "lineno", None)
                            end = getattr(stmt, "end_lineno", None)
                            if start and end:
                                return "\n".join(lines[start - 1 : end])
                            return ast.get_source_segment(source, stmt) or ""
                for child in node.body:
                    if isinstance(child, ast.ClassDef):
                        result = visit(child, stack)
                        if result:
                            return result
                return None
            finally:
                stack.pop()

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                result = visit(node, [])
                if result:
                    return result
        return ""

    def _collect_reference_tests(
        self,
        func: Dict[str, Any],
        call_graph: Dict[str, Any],
        test_lookup: Dict[str, Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Collect changed test snippets referencing this function."""
        func_name = func.get("name")
        if not func_name or not call_graph or not test_lookup:
            return []
        references: List[Dict[str, Any]] = []
        for caller_id, edges in call_graph.items():
            test_entry = test_lookup.get(caller_id)
            if not test_entry:
                continue
            matched_lines = [edge.get("line") for edge in (edges or []) if edge.get("callee") == func_name]
            if not matched_lines:
                continue
            snippet = (test_entry.get("source_after") or test_entry.get("source_before") or "").strip()
            references.append({
                "test_function_id": caller_id,
                "file": test_entry.get("file"),
                "call_lines": matched_lines,
                "source": snippet[:1500]
            })
            if len(references) >= limit:
                break
        return references

    def _collect_behavior_text(self, func: Dict[str, Any], oracle: Dict[str, Any]) -> str:
        """Aggregate source + oracle text for behavior dimension detection."""
        parts: List[str] = []

        for field in ("source_before", "source_after"):
            src = (func.get(field) or "")[:2000]
            if src:
                parts.append(src)

        # oracle: diff / must_preserve / may_change / behavior_notes
        for key in ("code_diff_summary", "must_preserve_region", "may_change_region", "behavior_notes"):
            items = oracle.get(key) or []
            if isinstance(items, list):
                parts.extend(str(x) for x in items)
            else:
                parts.append(str(items))

        cf = oracle.get("control_flow_hints") or {}
        if isinstance(cf, dict):
            for v in cf.values():
                if isinstance(v, list):
                    parts.extend(str(x) for x in v)
                else:
                    parts.append(str(v))

        return "\n".join(parts).lower()

    def _detect_behavior_dimensions(self, func: Dict[str, Any], oracle: Dict[str, Any]) -> Dict[str, bool]:
        """Detect high-level behavior dimensions (missing/size/order/boundary/...)."""
        text = self._collect_behavior_text(func, oracle)
        dims = {
            "missing_values": False,
            "size_boundary": False,
            "zero": False,
            "negative": False,
            "ordering": False,
            "boundary": False,
            "overflow_underflow": False,
            "monotonicity": False,
        }
        if not text.strip():
            return dims

        def has_any(tokens):
            return any(tok in text for tok in tokens if len(tok) >= 3)

        if has_any(MISSING_VALUE_TOKENS):
            dims["missing_values"] = True
        if has_any(SIZE_TOKENS):
            dims["size_boundary"] = True
        if has_any(ZERO_TOKENS):
            dims["zero"] = True
        if has_any(NEGATIVE_TOKENS):
            dims["negative"] = True
        if has_any(ORDERING_TOKENS):
            dims["ordering"] = True
        if has_any(BOUNDARY_TOKENS):
            dims["boundary"] = True
        if has_any(OVERFLOW_TOKENS):
            dims["overflow_underflow"] = True
        if has_any(MONOTONICITY_TOKENS):
            dims["monotonicity"] = True

        call_graph = self.impact_data.get("call_graph", {}) or {}
        func_id = func.get("function_id")
        for edge in call_graph.get(func_id, []):
            callee = (edge.get("callee") or "").lower()
            if callee in {"len", "__len__", "size", "repeat", "default_index"}:
                dims["size_boundary"] = True
            if callee in {"sort_values", "sort_index", "argsort", "nlargest", "nsmallest", "rank"}:
                dims["ordering"] = True

        return dims

    def build_view(self) -> Dict:
        """Build the target function view."""
        # Get PR-level intent
        pr_intent = self.intent_data.get('intent', {})

        # Build oracle lookup by function_id
        oracle_lookup = {}
        for oracle in self.oracle_data.get('function_oracles', []):
            func_id = oracle.get('function_id')
            if func_id:
                oracle_lookup[func_id] = oracle

        # Get changed functions and test references
        all_changed_functions = self.impact_data.get('changed_functions', []) or []
        changed_functions = [
            func for func in all_changed_functions
            if func.get('kind') == 'code'
        ]
        test_lookup = {
            func.get('function_id'): func
            for func in all_changed_functions
            if func.get('kind') == 'test' and func.get('function_id')
        }


        impact_funcs = [
            func for func in self.impact_data.get('impact_region', [])
            if func.get('kind') == 'code'
        ]

        all_code_funcs = {}
        for func in changed_functions:
            fid = func.get('function_id')
            if fid:
                all_code_funcs[fid] = func
        for func in impact_funcs:
            fid = func.get('function_id')
            if fid:
                all_code_funcs[fid] = func

        call_graph = self.impact_data.get('call_graph', {}) or {}

        neighbors_by_target: Dict[str, List[Dict]] = {}

        for func in changed_functions:
            func_id = func.get('function_id')
            func_name = func.get('name')
            if not func_id or not func_name:
                continue

            seen_ids = set()
            neighbors: List[Dict] = []
            callability = func.get("callability") or {}
            entry_kind = callability.get("entry_kind")
            class_name = callability.get("class_qualname")

            target_file = func.get('file')
            for edge in call_graph.get(func_id, []):
                callee_name = edge.get('callee')
                if not callee_name:
                    continue

                for nid, nf in all_code_funcs.items():
                    if nid == func_id:
                        continue
                    if nf.get('name') == callee_name and nid not in seen_ids:
                        callee_file = nf.get('file')
                        if target_file != callee_file:
                            target_module = _file_to_module(target_file)
                            callee_module = _file_to_module(callee_file or "")
                            if target_module != callee_module:
                                continue

                        neighbors.append({
                            'function_id': nid,
                            'role': 'callee',
                            'kind': 'code',
                            'name': nf.get('name'),
                            'qualified_name': nf.get('qualified_name'),
                            'file': nf.get('file'),
                            'span': nf.get('span') or nf.get('after_span'),
                            'signature': nf.get('signature', ''),
                            'callability': nf.get('callability', {}),
                        })
                        seen_ids.add(nid)

            for caller_id, edges in call_graph.items():
                if caller_id == func_id:
                    continue

                caller_func = all_code_funcs.get(caller_id)
                if not caller_func:
                    continue

                if any(e.get('callee') == func_name for e in edges):
                    caller_file = caller_func.get('file')
                    if target_file != caller_file:
                        caller_module = _file_to_module(caller_file or "")
                        target_module = _file_to_module(target_file)
                        if caller_module != target_module:
                            continue

                    if caller_id not in seen_ids:
                        neighbors.append({
                            'function_id': caller_id,
                            'role': 'caller',
                            'kind': 'code',
                            'name': caller_func.get('name'),
                            'qualified_name': caller_func.get('qualified_name'),
                            'file': caller_func.get('file'),
                            'span': caller_func.get('span') or caller_func.get('after_span'),
                            'signature': caller_func.get('signature', ''),
                            'callability': caller_func.get('callability', {}),
                        })
                        seen_ids.add(caller_id)

            if entry_kind == "class_field" and class_name:
                for nid, nf in all_code_funcs.items():
                    if nid == func_id or nid in seen_ids:
                        continue
                    n_callability = nf.get("callability") or {}
                    if n_callability.get("class_qualname") != class_name:
                        continue
                    n_entry_kind = n_callability.get("entry_kind")
                    if n_entry_kind not in {"method", "constructor", "property"}:
                        continue
                    neighbors.append({
                        "function_id": nid,
                        "role": "sibling_method",
                        "kind": nf.get("kind", "code"),
                        "name": nf.get("name"),
                        "qualified_name": nf.get("qualified_name"),
                        "file": nf.get("file"),
                        "span": nf.get("span") or nf.get("after_span"),
                        "signature": nf.get("signature", ""),
                        "callability": nf.get("callability", {}),
                    })
                    seen_ids.add(nid)

            neighbors_by_target[func_id] = neighbors

        # Build target function views
        target_functions = []
        for func in changed_functions:
            func_id = func.get('function_id')

            # Get oracle for this function
            oracle = oracle_lookup.get(func_id, {})

            # Detect behavior dimensions for this function
            behavior_dims = self._detect_behavior_dimensions(func, oracle)

            # Get neighbors for this specific function
            neighbors = neighbors_by_target.get(func_id, [])

            file_context = self._compute_stable_context(func.get('file'))
            source_text = file_context.get('source', '')
            callability = func.get('callability', {})
            init_signature = self._extract_init_signature(source_text, (callability or {}).get('class_qualname'))
            reference_tests = self._collect_reference_tests(func, call_graph, test_lookup)

            target_view = {
                'function_id': func_id,
                'name': func.get('name'),
                'qualified_name': func.get('qualified_name'),
                'file': func.get('file'),
                'before_span': func.get('before_span'),
                'after_span': func.get('after_span'),
                'source_before': func.get('source_before'),
                'source_after': func.get('source_after'),
                'callability': callability,
                'oracle': {
                    'code_diff_summary': oracle.get('code_diff_summary', []),
                    'setup_requirements': oracle.get('setup_requirements', []),
                    'must_preserve_region': oracle.get('must_preserve_region', []),
                    'may_change_region': oracle.get('may_change_region', []),
                    'behavior_notes': oracle.get('behavior_notes', []),
                    'control_flow_hints': oracle.get('control_flow_hints', {}),
                    'error': oracle.get('error'),
                    'skipped': oracle.get('skipped'),
                },
                'behavior_dimensions': behavior_dims,
                'impact_neighbors': neighbors,
                'code_context': {
                    'imports': "\n".join(file_context.get('imports', [])),
                    'init_signature': init_signature,
                },
                'reference_tests': reference_tests,
            }

            target_functions.append(target_view)

        # Build final output
        output = {
            'pr_number': self.pr_number,
            'pr_intent': pr_intent,
            'functions': target_functions
        }

        return output


def build_target_view(
    impact_region_path: str,
    function_oracle_path: Optional[str],
    pr_intent_path: Optional[str],
    output_path: str,
    repo_root: Optional[str] = None
) -> Dict:
    """
    Build target function view for a single PR.

    Args:
        impact_region_path: Path to impact region JSON
        function_oracle_path: Path to function oracles JSON
        pr_intent_path: Path to PR intent JSON
        output_path: Path to save output JSON

    Returns:
        Target view dictionary

    Example:
        result = build_target_view(
            'impact_region/output/marshmallow/pr_2699_impact.json',
            'function_oracle_generation/output/marshmallow/pr_2699_function_oracles.json',
            'intent_extraction/output/marshmallow/pr_2699_intent.json',
            'target_function_view/output/marshmallow/pr_2699_targets_view.json'
        )
    """
    # Build view
    builder = TargetFunctionViewBuilder(
        impact_region_path,
        function_oracle_path,
        pr_intent_path,
        repo_root
    )
    result = builder.build_view()

    # Save output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def process_pr(
    pr_number: int,
    repo_name: str,
    base_dir: str,
    output_dir: str
):
    """Process a single PR given its number and repository name."""
    base_path = Path(base_dir)

    # Construct input paths
    impact_path = base_path / 'impact_region' / 'output' / repo_name / f'pr_{pr_number}_impact.json'
    oracle_path = base_path / 'function_oracle_generation' / 'output' / repo_name / f'pr_{pr_number}_function_oracles.json'
    intent_path = base_path / 'intent_extraction' / 'output' / repo_name / f'pr_{pr_number}_intent.json'

    # Check if impact file exists (required)
    if not impact_path.exists():
        print(f"Error: Impact region file not found: {impact_path}", file=sys.stderr)
        return None

    # Check optional files
    oracle_path_str = str(oracle_path) if oracle_path.exists() else None
    intent_path_str = str(intent_path) if intent_path.exists() else None

    if not oracle_path_str:
        print(f"  Warning: Oracle file not found, using empty oracle: {oracle_path}")
    if not intent_path_str:
        print(f"  Warning: Intent file not found, using empty intent: {intent_path}")

    # Output path
    output_path = Path(output_dir) / repo_name / f'pr_{pr_number}_targets_view.json'

    # Resolve repo root for static context
    repo_root = None
    repo_candidates = [
        Path(base_dir) / 'project' / repo_name,
        Path(base_dir).parent / 'project' / repo_name
    ]
    for candidate in repo_candidates:
        if candidate.exists():
            repo_root = str(candidate)
            break
    if not repo_root:
        print(f"  Warning: Repository root not found for {repo_name}; code context will be limited.")

    print(f"Processing PR #{pr_number}...")
    result = build_target_view(
        str(impact_path),
        oracle_path_str,
        intent_path_str,
        str(output_path),
        repo_root
    )

    func_count = len(result.get('functions', []))
    print(f"  Generated view with {func_count} target function(s)")
    print(f"  Saved to {output_path}")

    return result


def main():
    """Main entry point for batch processing."""
    if len(sys.argv) < 3:
        print("Usage: build_target_view.py <base_dir> <repo_name> [pr_number1 pr_number2 ...]")
        print()
        print("Example:")
        print("  python3 build_target_view.py \\")
        print("    /path/to/ORION \\")
        print("    marshmallow \\")
        print("    2699 2022")
        print()
        print("For single PR:")
        print("  python3 -c 'from build_target_view import build_target_view; \\")
        print("    build_target_view(impact_path, oracle_path, intent_path, output_path)'")
        sys.exit(1)

    base_dir = sys.argv[1]
    repo_name = sys.argv[2]

    if len(sys.argv) > 3:
        # Process specific PRs
        pr_numbers = [int(pr) for pr in sys.argv[3:]]
    else:
        # Auto-discover PRs with all three input files
        print("Auto-discovering PRs with complete inputs...")
        base_path = Path(base_dir)
        impact_dir = base_path / 'impact_region' / 'output' / repo_name
        oracle_dir = base_path / 'function_oracle_generation' / 'output' / repo_name
        intent_dir = base_path / 'intent_extraction' / 'output' / repo_name

        # Find PRs that have impact files (oracle and intent are optional)
        pr_numbers = []
        if impact_dir.exists():
            for impact_file in sorted(impact_dir.glob('pr_*_impact.json')):
                pr_num = int(impact_file.stem.split('_')[1])
                pr_numbers.append(pr_num)

    if not pr_numbers:
        print(f"No PRs found with complete inputs for {repo_name}")
        sys.exit(1)

    print(f"Found {len(pr_numbers)} PRs to process: {pr_numbers}")
    print()

    output_dir = Path(base_dir) / 'target_function_view' / 'output'

    success_count = 0
    for pr_number in pr_numbers:
        try:
            result = process_pr(pr_number, repo_name, base_dir, str(output_dir))
            if result:
                success_count += 1
        except Exception as e:
            print(f"Error processing PR {pr_number}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print()
    print(f"Done! Successfully processed {success_count}/{len(pr_numbers)} PRs")
    print(f"Results saved to {output_dir / repo_name}")


if __name__ == '__main__':
    main()
