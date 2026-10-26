#!/usr/bin/env python3
"""
Impact Region Extractor v2 (rewritten)

Generates a concise JSON summary of the functions/class-fields impacted by a PR:
1. Changed functions/constructors/properties.
2. Class-level fields (e.g., Marshmallow schema fields, ORM columns).
3. Impact neighbors (callers/callees/sibling methods) with minimal context.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import resolve_project_checkout_from_base_dir

DEFAULT_BASE_DIR = os.environ.get("ORION_BASE_DIR", str(ROOT_DIR))


@dataclass
class FunctionSpan:
    """Holds source metadata for a function/method/property or class field."""

    name: str
    qualified_name: str
    file: str
    start_line: int
    end_line: int

    is_changed: bool = False
    entry_kind: str = "function"  # function | constructor | method | property | class_field
    class_qualname: Optional[str] = None
    is_staticmethod: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    control_flow_metrics: Dict[str, int] = field(default_factory=dict)


class DiffParser:
    """Parse unified diff text and return changed line ranges per file."""

    @staticmethod
    def parse(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
        changed_ranges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        current_file = None

        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                current_file = None
                m = re.match(r"^diff --git a/(.*?) b/(.*)$", line)
                if m:
                    current_file = m.group(2).strip().strip('"')
                continue

            if line.startswith("rename to "):
                path = line[len("rename to "):].strip().strip('"')
                if path and path != "/dev/null":
                    current_file = path
                continue

            if line.startswith("+++ "):
                p = line[4:].strip()
                if p.startswith("b/"):
                    path = p[2:].strip().strip('"')
                    if path and path != "/dev/null":
                        current_file = path
                continue

            if line.startswith("@@") and current_file:
                match = re.search(r"@@ -\d+,?\d* \+(\d+),?(\d*)", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    end = start + count - 1
                    changed_ranges[current_file].append((start, end))

        return dict(changed_ranges)


class ImpactAnalysisVisitor(ast.NodeVisitor):
    """
    AST visitor that captures changed functions/methods/properties and class fields.
    Also builds a lightweight call graph.
    """

    def __init__(self, file_path: str, changed_lines: Set[int], source_code: str):
        self.file_path = file_path
        self.changed_lines = changed_lines
        self.source_lines = source_code.splitlines()

        self.functions: List[FunctionSpan] = []
        self.class_fields: List[FunctionSpan] = []
        self.calls: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        self.current_class: List[str] = []
        self.current_function: Optional[str] = None
        self._function_depth = 0

    # Helpers -----------------------------------------------------------------

    def _intersects_diff(self, node: ast.AST) -> bool:
        if not self.changed_lines:
            return False
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None:
            return False
        span = set(range(start, (end or start) + 1))
        return bool(span & self.changed_lines)

    def _complexity_metrics(self, node: ast.AST) -> Dict[str, int]:
        metrics = {
            "branch_count": 0,
            "loop_count": 0,
            "try_count": 0,
            "pattern_match_count": 0,
            "conditional_expression_count": 0,
        }

        class FlowVisitor(ast.NodeVisitor):
            def visit_If(self, n): metrics["branch_count"] += 1; self.generic_visit(n)
            def visit_For(self, n): metrics["loop_count"] += 1; self.generic_visit(n)
            def visit_AsyncFor(self, n): metrics["loop_count"] += 1; self.generic_visit(n)
            def visit_While(self, n): metrics["loop_count"] += 1; self.generic_visit(n)
            def visit_Try(self, n): metrics["try_count"] += 1; self.generic_visit(n)
            def visit_IfExp(self, n): metrics["conditional_expression_count"] += 1; self.generic_visit(n)
            def visit_Match(self, n): metrics["pattern_match_count"] += 1; self.generic_visit(n)

        FlowVisitor().visit(node)
        metrics["complexity_score"] = (
            metrics["branch_count"]
            + metrics["conditional_expression_count"]
            + metrics["loop_count"] * 2
            + metrics["try_count"] * 2
            + metrics["pattern_match_count"] * 2
        )
        metrics["has_control_flow"] = int(
            any(
                metrics[key] > 0
                for key in (
                    "branch_count",
                    "conditional_expression_count",
                    "loop_count",
                    "try_count",
                    "pattern_match_count",
                )
            )
        )
        return metrics

    # Visitors ----------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current_class.append(node.name)
        self.generic_visit(node)
        self.current_class.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node: ast.AST, is_async: bool):
        class_qualname = ".".join(self.current_class) if self.current_class else None
        qualified_name = ".".join(self.current_class + [node.name])

        entry_kind = "function"
        if class_qualname:
            entry_kind = "constructor" if node.name == "__init__" else "method"
        if any(isinstance(dec, ast.Name) and dec.id == "property" for dec in node.decorator_list):
            entry_kind = "property"

        func_span = FunctionSpan(
            name=node.name,
            qualified_name=qualified_name,
            file=self.file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            is_changed=self._intersects_diff(node),
            entry_kind=entry_kind,
            class_qualname=class_qualname,
            is_staticmethod=any(isinstance(dec, ast.Name) and dec.id == "staticmethod" for dec in node.decorator_list),
            is_classmethod=any(isinstance(dec, ast.Name) and dec.id == "classmethod" for dec in node.decorator_list),
            is_property=(entry_kind == "property"),
            control_flow_metrics=self._complexity_metrics(node),
        )
        self.functions.append(func_span)

        prev_function = self.current_function
        self.current_function = qualified_name
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1
        self.current_function = prev_function

    def visit_Assign(self, node: ast.Assign):
        self._handle_field_definition(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._handle_field_definition(node)

    def _handle_field_definition(self, node: ast.AST):
        if not self.current_class or self._function_depth > 0:
            return
        if not self._intersects_diff(node):
            return

        field_names: List[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    field_names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.append(node.target.id)

        if not field_names:
            return

        class_name = ".".join(self.current_class)
        for fname in field_names:
            qualified = f"{class_name}.{fname}"
            span = FunctionSpan(
                name=fname,
                qualified_name=qualified,
                file=self.file_path,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                is_changed=True,
                entry_kind="class_field",
                class_qualname=class_name,
                control_flow_metrics={},
            )
            self.class_fields.append(span)

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            callee = self._get_call_name(node.func)
            if callee:
                self.calls[self.current_function].append((callee, getattr(node, "lineno", -1)))
        self.generic_visit(node)

    @staticmethod
    def _get_call_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


class StaticAnalyzer:
    """
    Static analyzer v2 for signature, default-value, exception, and return-type risks.
    """

    @staticmethod
    def _unparse(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except AttributeError:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, ast.Name):
                return node.id
            return "complex_expr"

    @staticmethod
    def _get_defaults_map(node: ast.AST) -> Dict[str, str]:
        defaults: Dict[str, str] = {}

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return defaults

        args = node.args.args
        defs = node.args.defaults
        if defs:
            offset = len(args) - len(defs)
            for idx, default_node in enumerate(defs):
                arg_name = args[offset + idx].arg
                defaults[arg_name] = StaticAnalyzer._unparse(default_node)

        for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if default_node is not None:
                defaults[arg.arg] = StaticAnalyzer._unparse(default_node)

        return defaults

    @staticmethod
    def _get_raised_exceptions(node: ast.AST) -> Set[str]:
        exceptions: Set[str] = set()
        if not node:
            return exceptions
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    exceptions.add(exc.func.id)
                elif isinstance(exc, ast.Name):
                    exceptions.add(exc.id)
        return exceptions

    @staticmethod
    def _is_generator(node: ast.AST) -> bool:
        if not node:
            return False
        return any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

    @staticmethod
    def analyze(source_before: str, source_after: str) -> Dict[str, Any]:
        if not source_before or not source_after:
            return {}

        try:
            tree_pre = ast.parse(source_before)
            tree_post = ast.parse(source_after)
        except SyntaxError:
            return {}

        func_pre = next((n for n in ast.walk(tree_pre) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        func_post = next((n for n in ast.walk(tree_post) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)

        if not func_pre or not func_post:
            return {}

        def collect_arg_info(node: ast.AST) -> Tuple[Set[str], bool]:
            args = set()
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return args, False
            args.update(a.arg for a in node.args.args)
            args.update(a.arg for a in node.args.kwonlyargs)
            has_kwargs = node.args.kwarg is not None
            return args, has_kwargs

        args_pre, _ = collect_arg_info(func_pre)
        args_post, kwargs_post = collect_arg_info(func_post)
        removed_args = sorted(args_pre - args_post)
        risk_silent_swallow = bool(removed_args and kwargs_post)

        defaults_pre = StaticAnalyzer._get_defaults_map(func_pre)
        defaults_post = StaticAnalyzer._get_defaults_map(func_post)
        changed_defaults: List[Dict[str, str]] = []
        for arg, old_val in defaults_pre.items():
            new_val = defaults_post.get(arg)
            if new_val is not None and new_val != old_val:
                changed_defaults.append({"arg": arg, "old": old_val, "new": new_val})

        exc_pre = StaticAnalyzer._get_raised_exceptions(func_pre)
        exc_post = StaticAnalyzer._get_raised_exceptions(func_post)
        exception_changes = {
            "removed": sorted(exc_pre - exc_post),
            "added": sorted(exc_post - exc_pre),
        }
        risk_exception_drift = bool(exception_changes["removed"] and exception_changes["added"])

        risk_generator_change = (not StaticAnalyzer._is_generator(func_pre)) and StaticAnalyzer._is_generator(func_post)

        return {
            "removed_args": removed_args,
            "risk_silent_swallow": risk_silent_swallow,
            "changed_defaults": changed_defaults,
            "exception_changes": exception_changes,
            "risk_exception_drift": risk_exception_drift,
            "risk_generator_change": risk_generator_change,
        }


class ImpactRegionExtractorV2:
    """Combine diff + AST analysis to produce final impact-region JSON."""

    MAX_FETCH_RETRIES = 3

    def __init__(self, repo_path: Path, pr_metadata: Dict[str, Any]):
        self.repo_path = Path(repo_path)
        self.pr_metadata = pr_metadata
        self.pre_commit = pr_metadata["pre_commit"]
        self.post_commit = pr_metadata["post_commit"]
        self.diff_text = pr_metadata.get("diff", "")
        self.MAX_FUNCTION_LINES = 80
        self._ensure_commits_available()

    # ------------------------------------------------------------------ Git IO

    def _has_git_object(self, sha: str) -> bool:
        result = subprocess.run(
            ["git", "cat-file", "-e", sha],
            cwd=self.repo_path,
            capture_output=True,
        )
        return result.returncode == 0

    def _fetch_commit(self, sha: str, depth: Optional[int]) -> bool:
        depth_flag = [f"--depth={depth}"] if depth else []
        for attempt in range(1, self.MAX_FETCH_RETRIES + 1):
            cmd = ["git", "fetch", *depth_flag, "origin", sha]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return True
            time.sleep(1)
        return False

    def _ensure_local_ref(self, sha: str):
        ref_name = f"refs/tags/pr_cache_{sha}"
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", ref_name],
            cwd=self.repo_path,
        )
        if result.returncode == 0:
            return
        subprocess.run(
            ["git", "update-ref", ref_name, sha],
            cwd=self.repo_path,
            capture_output=True,
        )

    def _ensure_commits_available(self):
        for commit in {self.pre_commit, self.post_commit}:
            if not commit:
                continue
            if self._has_git_object(commit):
                self._ensure_local_ref(commit)
                continue
            if not self._fetch_commit(commit, depth=1):
                if not self._fetch_commit(commit, depth=None):
                    raise RuntimeError(f"Unable to fetch commit {commit} in {self.repo_path}")
            self._ensure_local_ref(commit)

    # ------------------------------------------------------------------ Helpers

    def _get_file_content(self, file_path: str, commit: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "show", f"{commit}:{file_path}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def _file_to_module(file_path: str) -> str:
        module = file_path.replace("\\", "/").rstrip(".py").rstrip(".pyx")
        if module.endswith(".py"):
            module = module[:-3]
        parts = [segment for segment in module.split("/") if segment]
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[0] == "tests":
            parts = parts[1:]
        return ".".join(parts)

    def _make_function_id(self, file_path: str, qualified_name: str) -> str:
        module = self._file_to_module(file_path)
        if module:
            return f"{module}.{qualified_name}"
        return qualified_name

    def _determine_kind(self, file_path: str) -> str:
        normalized = file_path.lower()
        fname = file_path.split("/")[-1]
        if fname.startswith("test_") or fname.endswith("_test.py"):
            return "test"
        if "/tests/" in normalized or "/testing/" in normalized:
            return "test"
        if normalized.startswith("tests/") or normalized.startswith("testing/"):
            return "test"
        return "code"

    # ------------------------------------------------------------------ Parsing

    def extract(self) -> Dict[str, Any]:
        changed_ranges = DiffParser.parse(self.diff_text)
        raw_python_files = [
            f for f in self.pr_metadata.get("changed_files", [])
            if f.endswith((".py", ".pyx"))
        ]

        IGNORE_PREFIXES = (
            "examples/", "example/",
            "docs/", "doc/",
            "benchmarks/",
            "scripts/", "tools/",
        )

        def _is_library_file(path: str) -> bool:
            normalized = path.replace("\\", "/").lstrip("./")
            if normalized.startswith(IGNORE_PREFIXES):
                return False
            # Catch nested folders like foo/examples/bar.py
            if "/examples/" in normalized:
                return False
            return True

        python_files = [f for f in raw_python_files if _is_library_file(f)]

        if not python_files:
            if raw_python_files:
                print(f"  [Info] No library code changes found (filtered {len(raw_python_files)} files).")
            return {"pr_number": self.pr_metadata.get("pr_number"), "changed_functions": [], "impact_region": []}

        changed_nodes: List[FunctionSpan] = []
        all_nodes: Dict[str, FunctionSpan] = {}
        call_graph: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for file_path in python_files:
            file_ranges = changed_ranges.get(file_path, [])
            source_after = self._get_file_content(file_path, self.post_commit)
            if not source_after:
                continue

            line_set: Set[int] = set()
            for start, end in file_ranges:
                line_set.update(range(start, end + 1))

            try:
                tree = ast.parse(source_after)
            except SyntaxError:
                print(f"Warning: Could not parse {file_path}", file=sys.stderr)
                continue

            visitor = ImpactAnalysisVisitor(file_path, line_set, source_after)
            visitor.visit(tree)

            module_prefix = self._file_to_module(file_path)

            for func_span in visitor.functions:
                func_id = self._make_function_id(file_path, func_span.qualified_name)
                all_nodes[func_id] = func_span
                if func_span.is_changed:
                    changed_nodes.append(func_span)

            for field_span in visitor.class_fields:
                func_id = self._make_function_id(file_path, field_span.qualified_name)
                all_nodes[func_id] = field_span
                changed_nodes.append(field_span)

            for caller, callees in visitor.calls.items():
                caller_id = f"{module_prefix}.{caller}" if module_prefix else caller
                call_graph[caller_id].extend(callees)

        if not changed_nodes:
            return {"pr_number": self.pr_metadata.get('pr_number'), "changed_functions": [], "impact_region": []}

        class_method_map: Dict[str, List[FunctionSpan]] = defaultdict(list)
        for node in all_nodes.values():
            if node.class_qualname and node.entry_kind in {"method", "constructor", "property"}:
                class_method_map[node.class_qualname].append(node)

        impact_region = self._build_impact_region(changed_nodes, all_nodes, call_graph, class_method_map)
        return self._generate_output(changed_nodes, impact_region, call_graph)

    # ---------------------------------------------------------------- Neighbors

    def _build_impact_region(
        self,
        changed_nodes: List[FunctionSpan],
        all_nodes: Dict[str, FunctionSpan],
        call_graph: Dict[str, List[Tuple[str, int]]],
        class_method_map: Dict[str, List[FunctionSpan]],
    ) -> Dict[str, Dict[str, Any]]:
        impact: Dict[str, Dict[str, Any]] = {}
        changed_ids = {
            self._make_function_id(func.file, func.qualified_name)
            for func in changed_nodes
        }

        for func_id in list(changed_ids):
            if func_id not in call_graph:
                continue
            caller_func = all_nodes.get(func_id)
            target_file = caller_func.file if caller_func else ""
            target_module = self._file_to_module(target_file) if target_file else ""
            target_class = caller_func.class_qualname if caller_func else None

            for callee_name, line in call_graph[func_id]:
                matches = [
                    (candidate_id, candidate_func)
                    for candidate_id, candidate_func in all_nodes.items()
                    if candidate_func.name == callee_name and candidate_id not in changed_ids
                ]
                picked = self._pick_best_candidate(matches, target_file, target_module, target_class)
                if not picked:
                    continue
                candidate_id, candidate_func = picked
                if candidate_id not in impact:
                    impact[candidate_id] = {"function": candidate_func, "role": "callee", "call_lines": []}
                impact[candidate_id]["call_lines"].append(line)

        for changed_id, func in [(self._make_function_id(f.file, f.qualified_name), f) for f in changed_nodes]:
            caller_name = func.name
            target_file = func.file
            target_module = self._file_to_module(target_file)

            for caller_id, callees in call_graph.items():
                if caller_id in changed_ids:
                    continue
                matches = [line for callee_name, line in callees if callee_name == caller_name]
                if not matches:
                    continue

                caller_func = all_nodes.get(caller_id)
                if not caller_func:
                    continue

                caller_file = caller_func.file
                if caller_file != target_file:
                    caller_module = self._file_to_module(caller_file)
                    if caller_module != target_module:
                        continue

                if caller_id not in impact:
                    impact[caller_id] = {"function": caller_func, "role": "caller", "call_lines": []}
                impact[caller_id]["call_lines"].extend(matches)

        # --- Class field heuristics: if field changed, add sibling methods as neighbors
        for func in changed_nodes:
            if func.entry_kind != "class_field":
                continue
            class_name = func.class_qualname
            if not class_name:
                continue
            siblings = class_method_map.get(class_name, [])
            for sibling in siblings:
                sibling_id = self._make_function_id(sibling.file, sibling.qualified_name)
                if sibling_id in changed_ids or sibling_id in impact:
                    continue
                impact[sibling_id] = {"function": sibling, "role": "sibling_method", "call_lines": []}

        return impact

    def _pick_best_candidate(
        self,
        candidates: List[Tuple[str, FunctionSpan]],
        target_file: str,
        target_module: str,
        target_class: Optional[str],
    ) -> Optional[Tuple[str, FunctionSpan]]:
        if not candidates:
            return None

        if target_class:
            same_class = [
                cand for cand in candidates
                if getattr(cand[1], "class_qualname", None) == target_class
            ]
            if same_class:
                same_class_file = [c for c in same_class if c[1].file == target_file]
                if same_class_file:
                    return same_class_file[0]
                return same_class[0]

        same_file = [cand for cand in candidates if cand[1].file == target_file]
        if same_file:
            return same_file[0]

        same_module = [
            cand for cand in candidates
            if self._file_to_module(cand[1].file) == target_module
        ]
        if same_module:
            return same_module[0]
        return candidates[0]

    # ------------------------------------------------------------- Output JSON

    def _generate_output(
        self,
        changed_nodes: List[FunctionSpan],
        impact_region: Dict[str, Dict[str, Any]],
        call_graph: Dict[str, List[Tuple[str, int]]],
    ) -> Dict[str, Any]:
        output = {
            "pr_number": self.pr_metadata.get("pr_number"),
            "changed_functions": [],
            "impact_region": [],
            "call_graph": {},
        }

        for node in changed_nodes:
            func_id = self._make_function_id(node.file, node.qualified_name)
            kind = self._determine_kind(node.file)

            func_data = {
                "function_id": func_id,
                "role": "changed",
                "name": node.name,
                "qualified_name": node.qualified_name,
                "file": node.file,
                "kind": kind,
                "before_span": None,
                "after_span": [node.start_line, node.end_line],
                "callability": self._build_callability(node),
                "control_flow_metrics": node.control_flow_metrics or {},
            }

            after_source = self._get_file_content(node.file, self.post_commit)
            if after_source:
                func_data["source_after"] = self._extract_source_snippet(after_source, node.start_line, node.end_line)

            before_source = self._get_file_content(node.file, self.pre_commit)
            if before_source and node.entry_kind != "class_field":
                before_span = self._find_function_in_source(
                    before_source,
                    node.qualified_name,
                    node.name,
                )
            elif before_source and node.entry_kind == "class_field":
                before_span = self._find_field_in_source(before_source, node.class_qualname, node.name)
            else:
                before_span = None

            if before_source and before_span:
                func_data["before_span"] = list(before_span)
                func_data["source_before"] = self._extract_source_snippet(
                    before_source, before_span[0], before_span[1]
                )

            if (
                func_data.get("source_before")
                and func_data.get("source_after")
                and node.entry_kind != "class_field"
            ):
                analysis = StaticAnalyzer.analyze(func_data["source_before"], func_data["source_after"])
                if analysis:
                    func_data["static_analysis"] = analysis

            output["changed_functions"].append(func_data)

        for func_id, info in impact_region.items():
            node = info["function"]
            kind = self._determine_kind(node.file)
            entry = {
                "function_id": func_id,
                "role": info["role"],
                "name": node.name,
                "qualified_name": node.qualified_name,
                "file": node.file,
                "span": [node.start_line, node.end_line],
                "kind": kind,
                "callability": self._build_callability(node),
                "control_flow_metrics": node.control_flow_metrics or {},
            }

            source_after = self._get_file_content(node.file, self.post_commit)
            if source_after:
                entry["source_after"] = self._extract_source_snippet(source_after, node.start_line, node.end_line)
                entry["signature"] = self._extract_signature(source_after, node.start_line, node.end_line)
                if info.get("role") == "caller" and info.get("call_lines"):
                    entry["call_context"] = self._extract_call_context(source_after, info["call_lines"])

            output["impact_region"].append(entry)

        serializable_graph = {}
        for caller_id, callees in call_graph.items():
            serializable_graph[caller_id] = [
                {"callee": callee, "line": line}
                for callee, line in callees
            ]
        output["call_graph"] = serializable_graph
        return output

    # --------------------------------------------------------- Extract helpers

    @staticmethod
    def _extract_source_snippet(source: str, start: int, end: int) -> str:
        lines = source.splitlines()
        snippet = lines[start - 1 : end]
        if len(snippet) > 80:
            snippet = snippet[:80] + ["    # ... (truncated)"]
        return "\n".join(snippet)

    @staticmethod
    def _extract_signature(source: str, start: int, end: int) -> str:
        lines = source.splitlines()[start - 1 : end]
        if not lines:
            return ""
        signature = []
        in_signature = True
        docstring_lines = []
        docstring_open = None
        body_lines = []

        for line in lines:
            stripped = line.strip()
            if in_signature:
                signature.append(line)
                if stripped.endswith(":") and not stripped.endswith("\\"):
                    in_signature = False
                continue
            if docstring_open:
                docstring_lines.append(line)
                if docstring_open in stripped:
                    docstring_open = None
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_open = stripped[:3]
                docstring_lines.append(line)
                if stripped.count(docstring_open) >= 2:
                    docstring_open = None
                continue
            body_lines.append(line)
            if len(body_lines) >= 5:
                break

        snippet = signature + docstring_lines + body_lines
        return "\n".join(snippet)

    @staticmethod
    def _extract_call_context(source: str, call_lines: List[int], context: int = 3) -> List[Dict[str, Any]]:
        lines = source.splitlines()
        contexts = []
        for line_num in sorted(set(call_lines)):
            start = max(0, line_num - context - 1)
            end = min(len(lines), line_num + context)
            snippet = "\n".join(lines[start:end])
            contexts.append({"line": line_num, "snippet": snippet})
        return contexts

    def _find_function_in_source(self, source: str, qualified_name: str, func_name: str) -> Optional[Tuple[int, int]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        target = qualified_name
        fallback = None
        class_stack: List[str] = []

        class Finder(ast.NodeVisitor):
            def visit_ClassDef(self_inner, node: ast.ClassDef):
                class_stack.append(node.name)
                self_inner.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(self_inner, node: ast.FunctionDef):
                qname = ".".join(class_stack + [node.name])
                if qname == target:
                    raise StopIteration((node.lineno, getattr(node, "end_lineno", node.lineno)))
                nonlocal fallback
                if fallback is None and node.name == func_name:
                    fallback = (node.lineno, getattr(node, "end_lineno", node.lineno))
                self_inner.generic_visit(node)

        finder = Finder()
        try:
            finder.visit(tree)
        except StopIteration as exc:
            return exc.value
        return fallback

    def _find_field_in_source(self, source: str, class_name: Optional[str], field_name: str) -> Optional[Tuple[int, int]]:
        if not class_name:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        target_class = class_name.split(".")
        class_stack: List[str] = []
        result: Optional[Tuple[int, int]] = None

        class FieldFinder(ast.NodeVisitor):
            def visit_ClassDef(self_inner, node: ast.ClassDef):
                class_stack.append(node.name)
                if ".".join(class_stack) == class_name:
                    for stmt in node.body:
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                            targets = []
                            if isinstance(stmt, ast.Assign):
                                for t in stmt.targets:
                                    if isinstance(t, ast.Name):
                                        targets.append(t.id)
                            else:
                                if isinstance(stmt.target, ast.Name):
                                    targets.append(stmt.target.id)
                            if field_name in targets:
                                raise StopIteration(
                                    (stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno))
                                )
                self_inner.generic_visit(node)
                class_stack.pop()

        finder = FieldFinder()
        try:
            finder.visit(tree)
        except StopIteration as exc:
            result = exc.value
        return result

    def _build_callability(self, span: FunctionSpan) -> Dict[str, Any]:
        return {
            "entry_kind": span.entry_kind,
            "class_qualname": span.class_qualname,
            "is_staticmethod": span.is_staticmethod,
            "is_classmethod": span.is_classmethod,
            "is_property": span.is_property,
        }

    # ----------------------------------------------------------------- CLI API

def process_pr(pr_json_path: Path, repo_path: Path, output_path: Path) -> Dict[str, Any]:
    with open(pr_json_path, "r", encoding="utf-8") as fh:
        pr_metadata = json.load(fh)

    extractor = ImpactRegionExtractorV2(repo_path, pr_metadata)
    result = extractor.extract()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def build_impact_region(pr_json_path: str, repo_path: str, output_path: str) -> Dict[str, Any]:
    return process_pr(Path(pr_json_path), Path(repo_path), Path(output_path))


def main():
    parser = argparse.ArgumentParser(description="Build impact region for a single PR")
    parser.add_argument("--repo", help="Repository name (e.g., marshmallow)", required=False)
    parser.add_argument("--pr", type=int, help="PR number", required=False)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Base work directory")
    parser.add_argument("--metadata-dir", help="Directory containing PR metadata JSONs")
    parser.add_argument("--repo-path", help="Explicit path to the git repository")
    parser.add_argument("--clones-base", help="Compatibility flag (unused)", default=None)
    parser.add_argument("positional", nargs="*", help="Compatibility with legacy CLI: <base_dir> <repo> <pr>")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    repo = args.repo
    pr = args.pr

    if args.positional:
        if len(args.positional) != 3:
            parser.error("Legacy invocation requires exactly 3 positional arguments: <base_dir> <repo> <pr>")
        base_dir = Path(args.positional[0])
        repo = args.positional[1]
        pr = int(args.positional[2])

    if repo is None or pr is None:
        parser.error("Must provide --repo and --pr (or legacy positional arguments).")

    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else base_dir / "pr_metadata"
    metadata_file = metadata_dir / repo / f"pr_{pr}.json"
    if not metadata_file.exists():
        metadata_file = metadata_dir / repo / f"{pr}.json"
    if not metadata_file.exists():
        parser.error(f"PR metadata not found: {metadata_file}")

    if args.repo_path:
        repo_path = Path(args.repo_path)
    else:
        repo_path = resolve_project_checkout_from_base_dir(base_dir, repo)
    if not repo_path.exists():
        parser.error(f"Repository path not found: {repo_path}")

    output_path = base_dir / "impact_region" / "output" / repo / f"pr_{pr}_impact.json"

    print(f"Processing {repo} PR#{pr}...")
    print(f"  Metadata: {metadata_file}")
    print(f"  Repository: {repo_path}")
    print(f"  Output: {output_path}\n")

    result = process_pr(metadata_file, repo_path, output_path)
    changed_count = len(result.get("changed_functions", []))
    impact_count = len(result.get("impact_region", []))
    print(f"  ✓ Changed functions: {changed_count}")
    print(f"  ✓ Impact neighbors: {impact_count}")
    print(f"  ✓ Saved to {output_path}")


if __name__ == "__main__":
    main()
