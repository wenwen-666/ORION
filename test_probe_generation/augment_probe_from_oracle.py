#!/usr/bin/env python3
"""
Stage 4c: Oracle-driven probe augmentation
(Final Robust Version: Anti-Hallucination Import Fixer)
"""

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LLM_MODEL, OPENAI_COMPATIBLE_BASE_URL, build_openai_client_kwargs, resolve_llm_api_key
from openai import OpenAI

# Import shared code repair utilities
from code_repair_utils import (
    self_correct_probe as repair_probe_code,
    clean_llm_code,
)

BASE_DIR_DEFAULT = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))
GLOBAL_SPECS_CAP = 40
MAX_MUTATION_SPECS_PER_FUNC = 2
MAX_ORACLE_SPECS_PER_FUNC = 3


def compute_function_priority(function_data: Dict[str, Any]) -> int:
    oracle = function_data.get("oracle") or {}
    return len(oracle.get("may_change_region") or []) * 5 + len(oracle.get("must_preserve_region") or []) * 2


def extract_spec_list(spec_data: Any) -> List[Dict[str, Any]]:
    if isinstance(spec_data, dict):
        return spec_data.get("probes") or spec_data.get("specs") or []
    return spec_data or []


SYSTEM_PROMPT = """You are a Python QA Engineer specialized in Regression Testing.

Your goal: Generate SELF-CONTAINED, EXECUTABLE regression tests (Probes).

[CRITICAL REQUIREMENT: IMPORT CONSISTENCY]
The most common failure is "ImportError" due to hallucinated submodules.
1. **Source of Truth**: Use [Valid Imports from Source].
2. **Fallback**: If you need a helper symbol (e.g. `missing`, `utils`) NOT in Valid Imports, import it from the **LIBRARY ROOT**.
   - WRONG: `from my_lib.constants import missing` (Guessing submodule)
   - RIGHT: `from my_lib import missing` (Safe fallback)

[CRITICAL: HANDLING REMOVED FEATURES]
If testing a REMOVED parameter/feature:
1. **REPRODUCE THE ERROR**: Write code that uses the *OLD/REMOVED* parameter.
2. **DO NOT FIX**: Do NOT update the code to use the new API. We want to verify it fails (e.g. TypeError).

[Output Format]
Return ONLY a JSON object:
{
  "new_probes": [
    {
      "probe_code": "import my_lib\\n...",
      "oracle_ref": {"region": "may_change", "index": 0},
      "reason": "Reproducing removed parameter usage..."
    }
  ]
}
"""


class OracleSpecAugmenter:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = DEFAULT_LLM_MODEL):
        self.api_key = resolve_llm_api_key(api_key)
        # Configure this endpoint if you use a different OpenAI-compatible API provider.
        self.base_url = base_url or OPENAI_COMPATIBLE_BASE_URL
        self.model = model
        self.client = OpenAI(**build_openai_client_kwargs(api_key=self.api_key, base_url=self.base_url))

    def _clean_llm_code(self, code: Optional[str]) -> str:
        """
        Robustly extract executable code from LLM responses that may include markdown fences.
        """
        if not code:
            return ""

        text = str(code)
        fenced = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        else:
            generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
            if generic:
                text = generic.group(1)

        text = text.strip()
        lines = text.splitlines()

        if lines:
            first = lines[0].strip()
            if first and not first.startswith(("import", "from", "#", "def", "class")):
                cleaned = lines[1:]
                if cleaned:
                    text = "\n".join(cleaned).strip()

        return text

    def _fix_symbol_typos(self, code: str) -> str:
        """
        Normalize leaked internal variable names like `missing_` -> `missing`.
        Applies to import statements and bare symbol usages.
        """
        if not code:
            return ""

        fixed_lines = []
        for line in code.splitlines():
            # Skip strings that obviously contain literals
            if line.strip().startswith(("'", '"')):
                fixed_lines.append(line)
                continue

            if "import" in line:
                # Replace trailing underscores in imported names
                def replace_import(match: re.Match) -> str:
                    symbol = match.group(1)
                    return symbol[:-1] if symbol.endswith("_") else symbol

                line = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)_', lambda m: replace_import(m), line)

            # Replace standalone usages of sentinel names (missing_, etc.) outside strings
            line = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)_(?![A-Za-z0-9_])', r'\1', line)
            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2500,
            )
            text = clean_llm_code(resp.choices[0].message.content)
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as exc:
            print(f"[WARN] LLM call failed: {exc}")
            return {}

    def _normalize_new_probe(self, probe: Dict[str, Any], func: Dict[str, Any], suffix: int) -> Dict[str, Any]:
        if "probe_id" not in probe:
            clean_name = func.get("name", "unknown").replace("_", "")
            probe["probe_id"] = f"aug_{clean_name}_{suffix}"
        probe.setdefault("target_function_id", func.get("function_id"))
        probe.setdefault("type", "unit")

        raw_code = probe.get("probe_code") or probe.get("code") or probe.get("test_body")
        probe["probe_code"] = clean_llm_code(raw_code) if isinstance(raw_code, str) else ""

        if "oracle_region_kind" not in probe and probe.get("oracle_ref"):
            ref = probe["oracle_ref"]
            if isinstance(ref, dict):
                probe["oracle_region_kind"] = ref.get("region", "may_change")
        return probe

    # ------------------------------------------------------------------
    # Risk pattern strategies (4 patterns)
    # ------------------------------------------------------------------
    def _build_risk_pattern_strategy(self, func: Dict[str, Any]) -> str:
        analysis = func.get("static_analysis") or {}
        strategies: List[str] = []

        module_name = func.get("module", "")
        root_pkg = module_name.split(".")[0] if module_name else "the_library"

        removed_args = analysis.get("removed_args") or []
        if analysis.get("risk_silent_swallow") or removed_args:
            removed_list = ", ".join(removed_args) if removed_args else "the removed parameters"
            module_label = module_name or "this module"
            strategies.append(
                f"""
[🔥🔥 REGRESSION STRATEGY: REMOVED ARGUMENTS]
- DETECTED: Arguments [{removed_list}] were removed.
- **IMPORT ISSUE**: Sentinel values for these args (like `missing`) are likely removed from `{module_label}`.
- **ACTION**: Pass the removed arguments to reproduce the error, but do NOT try to import their sentinels from `{module_label}`.
- **FIX**: Import them from the **ROOT PACKAGE** `{root_pkg}` instead.
  - WRONG: `from {module_label} import missing`
  - RIGHT: `import {root_pkg}` then use `{root_pkg}.missing`
  - OR: `from {root_pkg}.utils import missing` (only if you are certain it exists)
"""
            )

        if analysis.get("changed_defaults"):
            strategies.append(
                """
[🔥🔥 REGRESSION STRATEGY: CHANGED DEFAULTS]
- DETECTED: Parameter default values changed.
- GOAL: Call the function WITHOUT supplying those parameters and assert the new default behavior.
"""
            )

        if analysis.get("risk_exception_drift"):
            strategies.append(
                """
[🔥🔥 REGRESSION STRATEGY: EXCEPTION CHANGE]
- DETECTED: Exception type/behavior changed.
- GOAL: Trigger the error condition.
- VERIFY: Assert that the NEW exception type is raised.
"""
            )

        if analysis.get("risk_generator_change"):
            strategies.append(
                """
[🔥🔥 REGRESSION STRATEGY: RETURN TYPE GENERATOR]
- DETECTED: Function return type changed to a Generator (yield).
- RISK: Consumers expecting a list/tuple will fail (e.g., `len()`, indexing `[0]`).
- GOAL: Call the function and inspect the return type. Verify if list operations are still supported or if consumption is required.
"""
            )

        return "\n".join(strategies).strip()

    # ------------------------------------------------------------------
    # Import sanitization & correction (Enhanced with Static Analysis)
    # ------------------------------------------------------------------
    def _parse_valid_symbols(self, valid_imports_block: str) -> Dict[str, str]:
        """
        Parse Valid Imports to build a mapping: symbol_name -> full_import_statement.

        Examples:
            'missing' -> 'from marshmallow.utils import missing'
            'Field' -> 'from marshmallow.fields import Field'
            'marshmallow' -> 'import marshmallow'
        """
        symbol_map: Dict[str, str] = {}
        if not valid_imports_block:
            return symbol_map

        for line in valid_imports_block.split('\n'):
            line = line.strip()
            if not line or not line.startswith(('import', 'from')):
                continue

            # Case 1: from X import Y, Z
            match_from = re.search(r'from\s+([\w\.]+)\s+import\s+(.+)', line)
            if match_from:
                module_path = match_from.group(1)
                symbols_str = match_from.group(2)

                for sym in symbols_str.split(','):
                    sym = sym.strip()
                    if not sym:
                        continue

                    # Handle "as alias"
                    if ' as ' in sym:
                        parts = sym.split(' as ')
                        original_name = parts[0].strip()
                        alias_name = parts[1].strip()
                        # Store both original and alias
                        symbol_map[original_name] = line
                        symbol_map[alias_name] = line
                    else:
                        symbol_map[sym] = line
                continue

            # Case 2: import X or import X.Y.Z
            match_import = re.search(r'import\s+([\w\.]+)(?:\s+as\s+([\w]+))?', line)
            if match_import:
                module_path = match_import.group(1)
                alias = match_import.group(2)

                # Store the top-level package
                top_pkg = module_path.split('.')[0]
                symbol_map[top_pkg] = line

                # Store the full path if it has dots
                if '.' in module_path:
                    symbol_map[module_path] = line

                # Store alias if present
                if alias:
                    symbol_map[alias] = line

        return symbol_map

    def _detect_used_symbols(self, code: str) -> set[str]:
        """
        Detect all symbols (identifiers) used in the code.
        Returns a set of symbol names that appear to be external references.
        """
        used_symbols: set[str] = set()

        # Extract all identifiers (word boundaries)
        all_tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', code)

        # Filter out Python keywords and builtins
        python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'try', 'while', 'with', 'yield'
        }

        python_builtins = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
            'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
            'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
            'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
            'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
            'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
            'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
            'super', 'tuple', 'type', 'vars', 'zip'
        }

        for token in all_tokens:
            if token not in python_keywords and token not in python_builtins:
                used_symbols.add(token)

        return used_symbols

    def _inject_missing_imports(self, code: str, symbol_map: Dict[str, str]) -> str:
        """
        Inject missing import statements for symbols that are used but not imported.
        Uses the symbol_map (from Valid Imports) as the source of truth.
        """
        if not symbol_map:
            return code

        # Detect which symbols are used in the code
        used_symbols = self._detect_used_symbols(code)

        # Extract currently imported symbols from the code
        imported_symbols: set[str] = set()
        for line in code.split('\n'):
            line = line.strip()

            # from X import Y, Z
            match_from = re.search(r'from\s+[\w\.]+\s+import\s+(.+)', line)
            if match_from:
                symbols_str = match_from.group(1)
                for sym in symbols_str.split(','):
                    sym = sym.strip()
                    if ' as ' in sym:
                        # "X as Y" - the alias Y is what's available
                        alias = sym.split(' as ')[1].strip()
                        imported_symbols.add(alias)
                    else:
                        imported_symbols.add(sym)

            # import X or import X as Y
            match_import = re.search(r'import\s+([\w\.]+)(?:\s+as\s+([\w]+))?', line)
            if match_import:
                module = match_import.group(1)
                alias = match_import.group(2)
                if alias:
                    imported_symbols.add(alias)
                else:
                    # Add the top-level package name
                    imported_symbols.add(module.split('.')[0])

        # Find symbols that need to be imported
        missing_imports: List[str] = []
        for symbol in used_symbols:
            # Skip if already imported
            if symbol in imported_symbols:
                continue

            # Check if this symbol is available in Valid Imports
            if symbol in symbol_map:
                import_stmt = symbol_map[symbol]
                # Avoid duplicates
                if import_stmt not in missing_imports and import_stmt not in code:
                    missing_imports.append(import_stmt)
                    print(f"    [Auto-Fix] Injecting missing import: {import_stmt}")

        # Inject at the top of the code
        if missing_imports:
            return '\n'.join(missing_imports) + '\n' + code

        return code

    def _sanitize_imports(self, code: str, root_pkg: str, valid_imports_block: str) -> str:
        if not root_pkg:
            return code

        valid_modules: set[str] = set()
        valid_symbols_map: Dict[str, set[str]] = defaultdict(set)
        for line in valid_imports_block.split('\n'):
            match = re.search(r'from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.+)', line)
            if match:
                module_name, symbols = match.groups()
                valid_modules.add(module_name)
                for symbol in symbols.split(','):
                    cleaned = symbol.strip()
                    if cleaned:
                        valid_symbols_map[module_name].add(cleaned.split()[0])
            else:
                fallback = re.search(r'from\s+([a-zA-Z0-9_\.]+)\s+import', line)
                if fallback:
                    valid_modules.add(fallback.group(1))

        new_lines: List[str] = []
        for raw in code.split('\n'):
            match = re.search(r'^(\s*)from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.+)', raw)
            if match:
                indent, module_path, symbols = match.groups()
                primary_symbol = ""
                stripped = symbols.strip()
                if stripped:
                    primary_symbol = stripped.split(',')[0].strip().split()[0]

                if module_path in valid_symbols_map and primary_symbol:
                    if primary_symbol not in valid_symbols_map[module_path]:
                        print(
                            f"    [Auto-Fix] Symbol '{primary_symbol}' not found in {module_path}, downgrading to {root_pkg}"
                        )
                        raw = f"{indent}from {root_pkg} import {primary_symbol}"
                elif module_path.startswith(root_pkg + '.') and module_path not in valid_modules:
                    print(f"    [Auto-Fix] Rewriting hallucinated import: {module_path} -> {root_pkg}")
                    raw = f"{indent}from {root_pkg} import {symbols}"
            new_lines.append(raw)
        return '\n'.join(new_lines)

    def _ensure_root_import(self, code: str, module_name: str) -> str:
        if not module_name:
            return code
        root_pkg = module_name.split('.')[0]
        has_usage = f"{root_pkg}." in code
        has_import = re.search(rf"^\s*import\s+{re.escape(root_pkg)}\b", code, re.MULTILINE)
        if has_usage and not has_import:
            return f"import {root_pkg}\n{code}"
        return code

    def _self_correct_probe(
        self,
        probe_code: str,
        source_after: str,
        valid_imports_block: str,
        module_name: str,
    ) -> str:
        """
        Apply static analysis-based code repair using shared utilities.
        Note: source_after parameter kept for backward compatibility but not used.
        """
        return repair_probe_code(probe_code, valid_imports_block, module_name)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_context_section(self, func: Dict[str, Any]) -> str:
        code_context = func.get("code_context") or {}
        oracle = func.get("oracle") or {}
        imports_block = code_context.get("imports", "")
        init_sig = code_context.get("init_signature", "")
        setup_reqs = oracle.get("setup_requirements") or []

        sections: List[str] = []
        if imports_block:
            sections.append(
                "[SAFE IMPORT ALLOWLIST]\n"
                "These imports exist in both Pre and Post versions. Prefer them before any fallback.\n"
                f"{imports_block}"
            )
        if init_sig:
            sections.append(f"[Constructor Signature]\n{init_sig}")
        if setup_reqs:
            sections.append("[Setup Requirements]\n" + "\n".join(f"- {req}" for req in setup_reqs))
        return "\n\n".join(sections)

    def _build_augment_prompt(
        self,
        func: Dict[str, Any],
        missing_items: List[Dict[str, Any]],
        max_new: int,
        existing_samples: List[Dict[str, Any]],
        propagated_risks: Optional[List[Dict]] = None,
    ) -> str:
        func_id = func["function_id"]
        source = (func.get("source_after") or "")[:2000]
        context_str = self._build_context_section(func)
        missing_text = "\n".join(f"- [{m['kind']}] {m['text']}" for m in missing_items)

        sample_code = ""
        if existing_samples:
            code = existing_samples[0].get("probe_code") or ""
            if code:
                sample_code = f"\n[Reference Probe]\n```python\n{code[:500]}...\n```"

        risk_strategy = self._build_risk_pattern_strategy(func)

        # Add propagated risks section
        risk_propagation_section = ""
        if propagated_risks:
            risk_lines = ["\n[PROPAGATED RISKS FROM CALLEES]"]
            risk_lines.append("This function calls other functions that have identified risks.")
            for risk in propagated_risks:
                risk_lines.append(f"- {risk['reason']}")
                if "test_input" in risk:
                    risk_lines.append(f"  Suggested test input: {risk['test_input']}")
            risk_propagation_section = "\n".join(risk_lines)

        return f"""
[Target Function]
ID: {func_id}
Module: {func.get('module', 'unknown')}
[Context]
{context_str}
[Source Snippet]
```python
{source}
```
[MISSING COVERAGE]
{missing_text}
{sample_code}
[Instruction]
Generate {max_new} NEW Probes.
Imports: If a symbol is missing from Valid Imports, try importing from the package root.
{risk_strategy}
{risk_propagation_section}
"""

    def _build_mutation_prompt(self, func: Dict[str, Any], max_new: int, existing_samples: List[Dict[str, Any]]) -> str:
        func_id = func["function_id"]
        context_str = self._build_context_section(func)
        samples_text = ""
        for idx, sample in enumerate(existing_samples[:2]):
            code = sample.get("probe_code") or ""
            samples_text += f"Sample {idx + 1}:\n```python\n{code[:600]}...\n```\n\n"

        risk_strategy = self._build_risk_pattern_strategy(func)

        return f"""
[Target Function]
ID: {func_id}
[Context]
{context_str}
[Samples]
{samples_text}
[Task]
Generate {max_new} NEW Probes by mutating samples.
Goal: Find regressions.
{risk_strategy}
"""

    def _llm_mutate_probes(self, func: Dict[str, Any], existing: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        prompt = self._build_mutation_prompt(func, limit, existing)
        resp = self._call_llm(SYSTEM_PROMPT, prompt)
        return resp.get("new_probes") or []

    # ------------------------------------------------------------------
    # Main augmentation pipeline
    # ------------------------------------------------------------------
    def augment_specs(self, target_view: Dict[str, Any], specs_data: Dict[str, Any], coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        existing_probes = extract_spec_list(specs_data)
        probes_by_func: Dict[str, List[Dict[str, Any]]] = {}
        for probe in existing_probes:
            tf_id = probe.get("target_function_id") or (probe.get("target_functions") or [None])[0]
            if tf_id:
                probes_by_func.setdefault(tf_id, []).append(probe)

        cov_by_func = {f["function_id"]: f for f in coverage_data.get("functions", [])}
        func_by_id = {f["function_id"]: f for f in target_view.get("functions", [])}

        remaining = GLOBAL_SPECS_CAP - len(existing_probes)
        if remaining <= 0:
            return specs_data

        print(f"[Augmenter] Budget: {remaining}")
        new_probes: List[Dict[str, Any]] = []

        # Mutation phase
        for func_id, func_probes in probes_by_func.items():
            if remaining <= 0:
                break
            func = func_by_id.get(func_id)
            if not func:
                continue

            if len(func_probes) >= 3:
                continue

            limit = min(MAX_MUTATION_SPECS_PER_FUNC, remaining)
            try:
                raw_list = self._llm_mutate_probes(func, func_probes, limit)
                valid_imports = func.get("code_context", {}).get("imports", "")
                module_name = func.get("module") or func.get("qualified_name", "").rpartition(".")[0]
                for raw in raw_list:
                    if remaining <= 0:
                        break
                    probe = self._normalize_new_probe(raw, func, len(new_probes))
                    if probe["probe_code"]:
                        probe["probe_code"] = self._self_correct_probe(
                            probe["probe_code"],
                            func.get("source_after", ""),
                            valid_imports,
                            module_name,
                        )
                    probe["source"] = "mutation"
                    new_probes.append(probe)
                    remaining -= 1
            except Exception as exc:
                print(f"Mutation err: {exc}")

        # Oracle-driven augmentation
        candidates = [
            (fid, cov.get("missing_items", []))
            for fid, cov in cov_by_func.items()
            if cov.get("missing_items")
        ]
        candidates.sort(key=lambda item: -len(item[1]))

        for func_id, missing_items in candidates:
            if remaining <= 0:
                break
            func = func_by_id.get(func_id)
            if not func:
                continue

            valid_imports = func.get("code_context", {}).get("imports", "")
            module_name = func.get("module") or func.get("qualified_name", "").rpartition(".")[0]
            limit = min(MAX_ORACLE_SPECS_PER_FUNC, remaining)

            try:
                existing = probes_by_func.get(func_id, [])
                prompt = self._build_augment_prompt(func, missing_items, limit, existing)
                resp = self._call_llm(SYSTEM_PROMPT, prompt)
                raw_list = resp.get("new_probes") or []
                for raw in raw_list:
                    if remaining <= 0:
                        break
                    probe = self._normalize_new_probe(raw, func, len(new_probes))
                    if probe["probe_code"]:
                        probe["probe_code"] = self._self_correct_probe(
                            probe["probe_code"],
                            func.get("source_after", ""),
                            valid_imports,
                            module_name,
                        )
                    probe["source"] = "oracle_augment"
                    new_probes.append(probe)
                    remaining -= 1
            except Exception as exc:
                print(f"Augment err: {exc}")

        print(f"[Augmenter] Added {len(new_probes)} new probes.")

        if isinstance(specs_data, dict):
            final = copy.deepcopy(specs_data)
            final["probes"] = existing_probes + new_probes
            final.pop("specs", None)
            return final
        return {"probes": existing_probes + new_probes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--base-dir", default=BASE_DIR_DEFAULT)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    base = Path(args.base_dir)
    target_view_path = base / "target_function_view" / "output" / args.repo / f"pr_{args.pr}_targets_view.json"
    specs_path = base / "test_probe_generation" / "output" / args.repo / f"pr_{args.pr}_test_specs.json"

    target_view = json.loads(target_view_path.read_text())
    specs = json.loads(specs_path.read_text())

    coverage_path = base / "test_probe_generation" / "output" / args.repo / f"pr_{args.pr}_oracle_coverage.json"
    if not coverage_path.exists():
        coverage_path = base / "test_spec_generation" / "output" / args.repo / f"pr_{args.pr}_oracle_coverage.json"
    coverage = json.loads(coverage_path.read_text())

    augmenter = OracleSpecAugmenter(api_key=args.api_key, base_url=args.base_url)
    new_specs = augmenter.augment_specs(target_view, specs, coverage)

    specs_path.write_text(json.dumps(new_specs, indent=2), encoding="utf-8")
    print(f"Updated {specs_path}")


if __name__ == "__main__":
    main()
