#!/usr/bin/env python3
"""
Probe Generator (formerly Spec Generator)

Generates self-contained "Probe" code blocks directly from targets_view.json.
Retains oracle_ref for coverage tracking, but delegates code logic entirely to LLM.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LLM_MODEL, OPENAI_COMPATIBLE_BASE_URL, build_openai_client_kwargs, resolve_llm_api_key
from openai import OpenAI
import openai

# Import shared code repair utilities
from code_repair_utils import self_correct_probe

PR_SPECS_MAX = 24
MIN_SPECS_PER_FUNC = 1
MAX_SPECS_PER_FUNC = 4
K_NEIGHBORS_PER_FUNC = 1
PR_NEIGHBOR_CAP = 5
MAX_SOURCE_CHARS = 2000
DEFAULT_BASE_DIR = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FunctionContext:
    """Holds metadata and generation results for a single changed function."""
    raw: Dict[str, Any]
    order: int
    module: str
    callable_name: str
    entry_callable: str
    priority: int
    entry_kind: str = "function"
    class_qualname: Optional[str] = None
    behavior_dimensions: Dict[str, bool] = field(default_factory=dict)
    is_pure_new: bool = False
    caller: Optional[Dict[str, Any]] = None
    unit_budget: int = 0
    integration_budget: int = 0
    probes: List[Dict[str, Any]] = field(default_factory=list)
    
    # [NEW] Fields for enhanced context
    setup_requirements: List[str] = field(default_factory=list)
    code_context: Dict[str, str] = field(default_factory=dict)
    static_analysis: Dict[str, Any] = field(default_factory=dict)


def load_targets_view(base_dir: str, repo: str, pr_number: int) -> Tuple[Dict[str, Any], Path]:
    path = (
        Path(base_dir)
        / "target_function_view"
        / "output"
        / repo
        / f"pr_{pr_number}_targets_view.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"targets_view not found: {path}")
    with open(path, "r") as fh:
        return json.load(fh), path


def module_from_file(file_path: str) -> str:
    if not file_path:
        return ""
    normalized = file_path.replace("\\", "/").lstrip("./")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[0] == "tests":
        parts = parts[1:]
    return ".".join(parts)


def is_code_function(function_data: Dict[str, Any]) -> bool:
    kind = function_data.get("kind") or function_data.get("node_kind")
    if kind:
        return kind == "code"
    return bool(function_data.get("source_after") or function_data.get("source_before"))


def is_changed_role(function_data: Dict[str, Any]) -> bool:
    role = function_data.get("function_role") or function_data.get("role")
    if role is None:
        return True
    if role.startswith("changed"):
        return True
    return role in {"changed_code", "changed_logic", "changed_interface", "changed_behavior"}


def compute_function_priority(function_data: Dict[str, Any]) -> int:
    oracle = function_data.get("oracle") or {}
    may = len(oracle.get("may_change_region") or [])
    must = len(oracle.get("must_preserve_region") or [])
    behavior = len(oracle.get("behavior_notes") or [])
    control = oracle.get("control_flow_hints") or {}
    new_paths = control.get("new_paths")
    if isinstance(new_paths, list):
        new_paths_count = len(new_paths)
    elif isinstance(new_paths, dict):
        new_paths_count = sum(len(v) for v in new_paths.values() if isinstance(v, list))
    else:
        new_paths_count = 0
    impact = len(function_data.get("impact_neighbors") or [])
    return may * 5 + new_paths_count * 3 + must * 2 + behavior + impact


def is_importable_entry(function: Dict[str, Any]) -> bool:
    callability = function.get("callability") or {}
    if callability.get("is_nested", False):
        return False
    fid = function.get("function_id") or ""
    if "<locals>" in fid:
        return False
    entry_kind = callability.get("entry_kind", "function")
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


def build_function_contexts(targets_view: Dict[str, Any]) -> List[FunctionContext]:
    contexts: List[FunctionContext] = []
    for idx, function in enumerate(targets_view.get("functions", [])):
        if not is_code_function(function):
            continue
        if not is_changed_role(function):
            continue

        oracle = function.get("oracle", {}) or {}
        if oracle.get("error"):
            continue
        if oracle.get("skipped"):
            print(f"  ⚠️  Skipping function whose oracle is marked as skipped: {function.get('function_id')}")
            continue

        if not is_importable_entry(function):
            print(f"  ⚠️  Skipping non-importable function: {function.get('function_id')}")
            continue

        file_path = function.get("file", "")
        module = module_from_file(file_path)
        if not module:
            continue

        function_id = function.get("function_id", "") or ""
        callability = function.get("callability") or {}
        entry_kind = callability.get("entry_kind", "function")
        class_qualname = callability.get("class_qualname")
        behavior_dimensions = function.get("behavior_dimensions") or {}

        if entry_kind == "constructor" and class_qualname:
            entry_callable = class_qualname.split(".")[-1]
        elif function_id and function_id.startswith(module + "."):
            entry_callable = function_id[len(module) + 1 :]
        elif function_id and module and module in function_id:
            idx_mod = function_id.rfind(module + ".")
            entry_callable = function_id[idx_mod + len(module) + 1 :]
        elif function_id:
             entry_callable = function_id.split(".", 1)[-1]
        else:
            entry_callable = Path(file_path).stem

        callable_name = entry_callable.split(".")[-1]
        source_before = (function.get("source_before") or "").strip()
        is_pure_new = not source_before or len(source_before) < 20
        priority = compute_function_priority(function)
        
        # [NEW] Load enhanced context from Target View
        setup_requirements = oracle.get("setup_requirements") or []
        code_context = function.get("code_context") or {}
        static_analysis = function.get("static_analysis") or {}

        contexts.append(
            FunctionContext(
                raw=function,
                order=idx,
                module=module,
                callable_name=callable_name,
                entry_callable=entry_callable,
                priority=priority,
                entry_kind=entry_kind,
                class_qualname=class_qualname,
                behavior_dimensions=behavior_dimensions,
                is_pure_new=is_pure_new,
                setup_requirements=setup_requirements,
                code_context=code_context,
                static_analysis=static_analysis
            )
        )
    return contexts


def neighbor_priority(ctx: FunctionContext, neighbor: Dict[str, Any]) -> Tuple[int, int]:
    target_file = ctx.raw.get("file")
    neighbor_file = neighbor.get("file")
    if neighbor_file == target_file:
        return (0, 0)
    neighbor_module = module_from_file(neighbor.get("file", ""))
    if neighbor_module and neighbor_module == ctx.module:
        return (1, 0)
    return (2, 0)


def select_integration_entries(contexts: List[FunctionContext]) -> int:
    from collections import defaultdict
    caller_map: Dict[str, List[Tuple[FunctionContext, Dict[str, Any], Tuple[int, int]]]] = defaultdict(list)

    for ctx in contexts:
        for neighbor in ctx.raw.get("impact_neighbors", []):
            if neighbor.get("role") != "caller":
                continue
            if neighbor.get("kind") not in (None, "code"):
                continue
            n_callability = neighbor.get("callability") or {}
            if n_callability.get("entry_kind", "function") in ("method", "property"):
                continue
            neighbor_id = neighbor.get("function_id")
            if not neighbor_id:
                continue
            priority_tuple = neighbor_priority(ctx, neighbor)
            caller_map[neighbor_id].append((ctx, neighbor, priority_tuple))

    must_generate = []
    optional = []

    for caller_id, items in caller_map.items():
        if len(items) >= 2:
            must_generate.append((caller_id, items))
        else:
            optional.append((caller_id, items))

    must_generate.sort(key=lambda item: (-len(item[1]), -max(c.priority for c, _, _ in item[1])))
    optional.sort(key=lambda item: (-item[1][0][0].priority))

    unique_callers_selected = 0
    assigned_caller_ids: set[str] = set()

    for caller_id, items in must_generate:
        if unique_callers_selected >= PR_NEIGHBOR_CAP: break
        for ctx, neighbor, _ in items:
            ctx.caller = neighbor
        assigned_caller_ids.add(caller_id)
        unique_callers_selected += 1

    for caller_id, items in optional:
        if unique_callers_selected >= PR_NEIGHBOR_CAP: break
        ctx, neighbor, _ = items[0]
        ctx.caller = neighbor
        assigned_caller_ids.add(caller_id)
        unique_callers_selected += 1

    return unique_callers_selected


def assign_budgets(contexts: List[FunctionContext]) -> None:
    if not contexts: return
    ordered = sorted(contexts, key=lambda c: (-c.priority, c.order))
    pure_new_funcs = [ctx for ctx in ordered if ctx.is_pure_new]
    modified_funcs = [ctx for ctx in ordered if not ctx.is_pure_new]
    remaining = PR_SPECS_MAX

    for ctx in ordered:
        ctx.unit_budget = 0
        ctx.integration_budget = 0

    # Round 0: Pure new with caller
    pure_new_with_caller = [ctx for ctx in pure_new_funcs if ctx.caller]
    pure_new_with_caller.sort(key=lambda c: (-c.priority, c.order))
    for ctx in pure_new_with_caller:
        if remaining <= 0: break
        ctx.integration_budget = 1
        remaining -= 1

    if remaining <= 0: return

    # Round 1: Modified -> 1 unit
    for ctx in modified_funcs:
        if remaining <= 0: break
        ctx.unit_budget = 1
        remaining -= 1

    if remaining <= 0: return

    # Round 2: Modified -> 2 unit
    for ctx in modified_funcs:
        if remaining <= 0: break
        if ctx.unit_budget == 1:
            ctx.unit_budget += 1
            remaining -= 1

    if remaining <= 0: return

    # Round 3: Modified with caller -> 1 integration
    mod_with_caller = [ctx for ctx in modified_funcs if ctx.caller]
    mod_with_caller.sort(key=lambda c: (-c.priority, c.order))
    for ctx in mod_with_caller:
        if remaining <= 0: break
        if ctx.integration_budget == 0:
            ctx.integration_budget = 1
            remaining -= 1

    if remaining <= 0: return

    # Round 4: Pure new -> 1 unit
    for ctx in pure_new_funcs:
        if remaining <= 0: break
        if ctx.unit_budget == 0:
            ctx.unit_budget = 1
            remaining -= 1

    if remaining <= 0: return

    # Round 5: Fill remaining
    for ctx in ordered:
        if remaining <= 0: break
        total = ctx.unit_budget + ctx.integration_budget
        if total >= MAX_SPECS_PER_FUNC: continue
        can_add = min(MAX_SPECS_PER_FUNC - total, remaining)
        if can_add > 0:
            ctx.unit_budget += can_add
            remaining -= can_add


def truncate_source(source: Optional[str]) -> str:
    if not source: return ""
    source = source.strip()
    if len(source) <= MAX_SOURCE_CHARS: return source
    return source[:MAX_SOURCE_CHARS] + "\n# ... truncated ..."


def module_and_callable_from_neighbor(neighbor: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not neighbor: return "", ""
    module = module_from_file(neighbor.get("file", ""))
    name = neighbor.get("name", "")
    return module, name


# ==============================================================================
# ==============================================================================

def _build_risk_pattern_strategy(ctx: FunctionContext) -> str:
    """
    Build mandatory probe strategy snippets based on static analysis risk tags.
    """
    analysis = ctx.static_analysis or ctx.raw.get("static_analysis") or {}
    strategies: List[str] = []

    removed_args = analysis.get("removed_args") or []
    if analysis.get("risk_silent_swallow") or removed_args:
        removed_list = ", ".join(removed_args) if removed_args else "the removed parameters"
        sentinel_hint = ctx.module.split(".")[0] if ctx.module else "the library root"
        strategies.append(
            f"""
[🔥🔥 REGRESSION STRATEGY: REMOVED ARGUMENTS]
- DETECTED: Arguments [{removed_list}] were removed.
- GOAL: Explicitly pass these removed arguments to ensure the new behavior (TypeError/warning) is enforced.
- EXPECTATION: The call should fail loudly; do NOT fix the invocation.
- SENTINEL IMPORTS: If these args relied on sentinels (e.g. `missing`) that vanished from this module, import them from `{sentinel_hint}` or its utils module.
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
- GOAL: Trigger the error condition and assert the NEW exception type or message.
"""
        )

    if analysis.get("risk_generator_change"):
        strategies.append(
            """
[🔥🔥 REGRESSION STRATEGY: RETURN TYPE GENERATOR]
- DETECTED: Function return type changed to a Generator (yield).
- RISK: Consumers expecting a list/tuple will fail (e.g., len(), indexing [0]).
- GOAL:
  1. Call the function.
  2. Verify whether the return value is now a generator/iterator.
  3. Attempt list-like operations (len, indexing) or consume the generator to expose potential breakage.
"""
        )

    return "\n".join(strategies).strip()


def build_code_diff_prompt(ctx: FunctionContext, pr_intent: Dict[str, Any]) -> str:
    function_id = ctx.raw.get("function_id", "")
    pr_intent_json = json.dumps(pr_intent or {}, indent=2)
    is_field = ctx.entry_kind == "class_field"
    
    # [NEW] Prepare enhanced context strings
    imports_block = ctx.code_context.get("imports", "")
    if isinstance(imports_block, list):
        imports_block = "\n".join(imports_block)
    elif imports_block is None:
        imports_block = ""
    init_sig = ctx.code_context.get("init_signature", "")
    setup_reqs = "\n".join(f"- {req}" for req in ctx.setup_requirements)
    
    context_section = ""
    if imports_block or init_sig or setup_reqs:
        context_section_lines = ["[Instantiation & Imports Context]", "Use this to fix imports and initialization:"]
        if imports_block:
            context_section_lines.append("Imports used in file:")
            context_section_lines.append(imports_block)
        if init_sig:
            context_section_lines.append("Constructor Signature:")
            context_section_lines.append(init_sig)
        if setup_reqs:
            context_section_lines.append("Setup Instructions (from existing tests):")
            context_section_lines.append(setup_reqs)
        context_section = "\n".join(context_section_lines) + "\n"

    if is_field:
        execution_instruction = (
            "3. **EXECUTION Strategy for Class Field**:\n"
            f"   - This target is a FIELD definition on `{ctx.class_qualname}`. Do not call it like a function.\n"
            f"   - Instantiate `{ctx.class_qualname}` using the constructor context above.\n"
            f"   - Access the attribute `{ctx.callable_name}` on the instance and verify behavior.\n"
        )
    else:
        execution_instruction = "3. **EXECUTION**: Call the function/method with arguments that trigger the changed lines.\n"

    risk_strategy = _build_risk_pattern_strategy(ctx)

    return (
        "You are a Python QA Engineer.\n"
        "Write a SELF-CONTAINED Probe to exercise changes.\n\n"
        "[Global PR intent]\n"
        f"{pr_intent_json}\n\n"
        "[Target info]\n"
        f"- ID: {function_id}\n"
        f"- Type: {ctx.entry_kind}\n"
        f"{'- Class: ' + ctx.class_qualname if ctx.class_qualname else ''}\n\n"
        "[Source BEFORE]\n"
        f"{truncate_source(ctx.raw.get('source_before'))}\n\n"
        "[Source AFTER]\n"
        f"{truncate_source(ctx.raw.get('source_after'))}\n\n"
        f"{context_section}\n"
        "[Instruction]\n"
        "Write ONE Probe that triggers the CHANGED lines in [Source AFTER].\n\n"
        "[LOGGING & OBSERVABILITY RULES - CRITICAL]\n"
        "1. **PRINT EVERYTHING**: Do not use silent assertions (like `assert`). Instead, use `print()` to output the result.\n"
        "2. **DUMP STATE**: When an object is created, print its relevant attributes. Example: `print(f'field.required={f.required}')`.\n"
        "3. **TRACE EXECUTION**: Print markers like 'Step 1 complete', 'Step 2 complete'.\n"
        "4. **CATCH & PRINT**: If testing for an exception, catch it and print the error message explicitly.\n"
        "   Example:\n"
        "   try:\n"
        "       code_that_fails()\n"
        "   except TypeError as e:\n"
        "       print(f'CAUGHT EXPECTED ERROR: {e}')\n"
        "   else:\n"
        "       print('NO ERROR RAISED (Unexpected)')\n\n"
        "[OBSERVABILITY RULES]\n"
        "1. NO SILENT SUCCESS: The probe MUST print detailed execution info to stdout.\n"
        "2. INSPECT OBJECTS: After initializing an object, print its key attributes (e.g. `print(vars(obj))`).\n"
        "3. EXPLICIT EXCEPTIONS: Use try/except blocks. Print 'Caught expected <Type>: <Msg>' on success, or 'Did not catch expected error' on failure.\n\n"
        "[SAFETY & RESOURCE RULES - CRITICAL]\n"
        "1. **NO PROGRAM TERMINATION**: NEVER use `sys.exit()`, `os._exit()`, or `quit()`. Let the probe complete naturally.\n"
        "2. **NO INFINITE LOOPS**: Do not write `while True:` or unbounded loops. Use finite iterations only.\n"
        "3. **NO HEAVY OPERATIONS**: Avoid network requests, large file I/O, or operations that take >1 second.\n"
        "4. **NO RESOURCE ABUSE**: Do not allocate large amounts of memory (>100MB) or create many objects.\n"
        "5. **BOUNDED EXECUTION**: Keep probe execution under 5 seconds. Use simple, focused tests.\n\n"
        "Rules:\n"
        "1. **SELF-CONTAINED**: Write all imports explicitly.\n"
        f"2. **IMPORTS**: Use `import {ctx.module}` and the context above.\n"
        "3. **SETUP**: Initialize instances using the provided Constructor Signature and Setup Instructions.\n"
        f"{execution_instruction}"
        f"{risk_strategy + chr(10) if risk_strategy else ''}"
        "4. **NO ASSERTION CRASHES**: Print results or use assertions.\n\n"
        "STRICT FORMAT RULE:\n"
        "- Do NOT wrap code in markdown blocks (e.g. ```python ... ```).\n"
        "- Do NOT include comments before imports.\n"
        "- Start the response DIRECTLY with import statements.\n\n"
        "Output JSON:\n"
        "{\n"
        '  "probes": [\n'
        '    {\n'
        '      "probe_id": "diff_cover_01",\n'
        '      "type": "unit",\n'
        '      "oracle_region_tag": "code_diff_coverage",\n'
        '      "probe_code": "import ...\\n\\n# Setup\\n..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def build_llm_prompt(
    ctx: FunctionContext,
    pr_intent: Dict[str, Any],
    unit_budget: int,
    integration_budget: int,
    existing_probes: Optional[List[Dict[str, Any]]] = None,
) -> str:
    function_id = ctx.raw.get("function_id", "")
    oracle = ctx.raw.get("oracle") or {}
    must_list = oracle.get("must_preserve_region") or []
    may_list = oracle.get("may_change_region") or []

    neighbor_text = ""
    neighbors = ctx.raw.get("impact_neighbors", [])
    if neighbors:
        n_list = []
        for n in neighbors[:5]:
            role = n.get("role", "related")
            name = n.get("name", "")
            n_list.append(f"- {name} ({role})")
        neighbor_text = "\n[Related Methods]\n" + "\n".join(n_list)

    # [NEW] Prepare enhanced context strings
    imports_block = ctx.code_context.get("imports", "")
    if isinstance(imports_block, list):
        imports_block = "\n".join(imports_block)
    elif imports_block is None:
        imports_block = ""
    init_sig = ctx.code_context.get("init_signature", "")
    setup_reqs = "\n".join(f"- {req}" for req in ctx.setup_requirements)

    safe_imports_text = (imports_block or "").strip()
    init_sig_text = (init_sig or "").strip()
    setup_text = setup_reqs.strip()
    context_section = ""
    if safe_imports_text or init_sig_text or setup_text:
        context_section = f"""
[SAFE IMPORT ALLOWLIST]
The following imports are guaranteed to exist in BOTH old and new versions.
**PRIORITY**: Use these paths first.
{safe_imports_text or '(none)'}

[Constructor Signature]
{init_sig_text or '(not provided)'}

[Setup Instructions]
{setup_text or '(none)'}
""".strip("\n")

    is_field = ctx.entry_kind == "class_field"
    if is_field:
        target_guide = (
            "[TARGET IS A CLASS FIELD]\n"
            f"- Target: `{ctx.callable_name}` in class `{ctx.class_qualname}`.\n"
            f"- Instantiate `{ctx.class_qualname}` using the constructor info above.\n"
            f"- Verify attribute `{ctx.callable_name}` and related behavior.\n"
        )
    else:
        caller_info = ""
        if ctx.caller and integration_budget > 0:
            c_mod, c_call = module_and_callable_from_neighbor(ctx.caller)
            caller_info = f"  - For integration, call via {c_mod}.{c_call}\n"
        target_guide = (
            "[TARGET IS A FUNCTION/METHOD]\n"
            "1. Instantiate class (if method) using Context info.\n"
            f"2. Call `{ctx.callable_name}` directly.\n"
            f"{caller_info}"
        )

    risk_strategy = _build_risk_pattern_strategy(ctx)
    root_pkg = ctx.module.split(".")[0] if ctx.module else "the_library"

    return (
        "You are a Python QA Engineer.\n"
        "Write SELF-CONTAINED Probes based on the Oracle.\n\n"
        "[Target Function]\n"
        f"- ID: {function_id}\n"
        f"- Module: {ctx.module}\n"
        f"- Type: {ctx.entry_kind}\n\n"
        "[Oracle (Behavior Contract)]\n"
        "Must Preserve:\n"
        f"{chr(10).join(f'  [{i}] {item}' for i, item in enumerate(must_list)) if must_list else '  - (none)'}\n\n"
        "May Change:\n"
        f"{chr(10).join(f'  [{i}] {item}' for i, item in enumerate(may_list)) if may_list else '  - (none)'}\n\n"
        f"{neighbor_text}\n"
        f"{context_section}\n"
        f"{risk_strategy + chr(10) if risk_strategy else ''}"
        "[Budgets]\n"
        f"- Unit Probes: {unit_budget}\n"
        f"- Integration Probes: {integration_budget}\n\n"
        f"{target_guide}\n"
        "[CRITICAL RULES]\n"
        "1. **SELF-CONTAINED**: Code must run entirely on its own.\n"
        "2. **IMPORTS**:\n"
        "   - Prefer imports from [SAFE IMPORT ALLOWLIST].\n"
        f"   - **FALLBACK**: If a symbol is missing there, import it from `{root_pkg}` (the library root).\n"
        "   - DO NOT guess submodules that are not in the allowlist.\n"
        "3. **SETUP**: FOLLOW [Setup Instructions] strictly.\n"
        "4. **VALIDITY**: Prefer real objects created from the provided context.\n\n"
        "[LOGGING & OBSERVABILITY RULES - CRITICAL]\n"
        "1. **PRINT EVERYTHING**: Do not use silent assertions (like `assert`). Instead, use `print()` to output the result.\n"
        "2. **DUMP STATE**: When an object is created, print its relevant attributes. Example: `print(f'field.required={f.required}')`.\n"
        "3. **TRACE EXECUTION**: Print markers like 'Step 1 complete', 'Step 2 complete'.\n"
        "4. **CATCH & PRINT**: If testing for an exception, catch it and print the error message explicitly.\n"
        "   Example:\n"
        "   try:\n"
        "       code_that_fails()\n"
        "   except TypeError as e:\n"
        "       print(f'CAUGHT EXPECTED ERROR: {e}')\n"
        "   else:\n"
        "       print('NO ERROR RAISED (Unexpected)')\n\n"
        "[OBSERVABILITY RULES]\n"
        "1. NO SILENT SUCCESS: The probe MUST print detailed execution info to stdout.\n"
        "2. INSPECT OBJECTS: After initializing an object, print its key attributes (e.g. `print(vars(obj))`).\n"
        "3. EXPLICIT EXCEPTIONS: Use try/except blocks. Print 'Caught expected <Type>: <Msg>' on success, or 'Did not catch expected error' on failure.\n\n"
        "[SAFETY & RESOURCE RULES - CRITICAL]\n"
        "1. **NO PROGRAM TERMINATION**: NEVER use `sys.exit()`, `os._exit()`, or `quit()`. Let the probe complete naturally.\n"
        "2. **NO INFINITE LOOPS**: Do not write `while True:` or unbounded loops. Use finite iterations only.\n"
        "3. **NO HEAVY OPERATIONS**: Avoid network requests, large file I/O, or operations that take >1 second.\n"
        "4. **NO RESOURCE ABUSE**: Do not allocate large amounts of memory (>100MB) or create many objects.\n"
        "5. **BOUNDED EXECUTION**: Keep probe execution under 5 seconds. Use simple, focused tests.\n\n"
        "STRICT FORMAT RULE:\n"
        "- Do NOT wrap code in markdown blocks (e.g. ```python ... ```).\n"
        "- Do NOT invent constants like `UNEXPECTED` or `Step` as importable symbols.\n"
        "- Start the response DIRECTLY with import statements.\n\n"
        "Output JSON:\n"
        "{\n"
        '  "probes": [\n'
        "    {\n"
        '      "type": "unit",\n'
        '      "oracle_ref": {"region": "must_preserve", "index": 0},\n'
        '      "probe_code": "..." \n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def extract_json_object(text: str) -> Optional[str]:
    if not text: return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None

def ensure_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else ([value] if value else [])

def deduplicate_probes(
    ctx: FunctionContext,
    raw_response: Dict[str, Any],
    global_seen_ids: Set[str],
    unit_cap: Optional[int] = None,
    integration_cap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    raw_probes = ensure_list(raw_response.get("probes")) or \
                 ensure_list(raw_response.get("unit_specs")) + ensure_list(raw_response.get("integration_specs"))

    valid_probes = []
    seen_code = set()
    
    unit_cap = float("inf") if unit_cap is None else unit_cap
    integration_cap = float("inf") if integration_cap is None else integration_cap

    count = 0
    unit_used = 0
    int_used = 0
    for p in raw_probes:
        code = p.get("probe_code") or p.get("test_body")
        if not code or len(code) < 10: continue

        # Apply static analysis-based code repair
        valid_imports = ctx.code_context.get('imports', '')
        code = self_correct_probe(code, valid_imports, ctx.module)

        if code in seen_code: continue
        seen_code.add(code)
        
        ptype = (p.get("type") or "unit").lower()
        clean_name = ctx.callable_name.replace("_", "")
        while True:
            pid = f"probe_{clean_name}_{ptype}_{count}"
            if pid not in global_seen_ids:
                global_seen_ids.add(pid)
                break
            count += 1
        
        oref = p.get("oracle_ref")
        region_kind = "may_change"
        if isinstance(oref, dict) and oref.get("region") == "must_preserve":
            region_kind = "must_preserve"
            
        if ptype == "integration":
            if int_used >= integration_cap: continue
        else:
            if unit_used >= unit_cap: continue

        valid_probes.append({
            "probe_id": pid,
            "type": ptype,
            "probe_code": code,
            "oracle_ref": oref,
            "oracle_region_kind": region_kind,
            "target_function_id": ctx.raw.get("function_id")
        })
        if ptype == "integration": int_used += 1
        else: unit_used += 1
        if unit_used >= unit_cap and int_used >= integration_cap: break

    return valid_probes

def _count_probes_by_type(probes: List[Dict[str, Any]]) -> Tuple[int, int]:
    unit = 0
    integration = 0
    for probe in probes:
        if (probe.get("type") or "unit").lower() == "integration":
            integration += 1
        else:
            unit += 1
    return unit, integration

def call_llm_with_retry(client: OpenAI, max_retries: int = 3, **kwargs) -> Any:
    """Call the LLM with retry support.

    Args:
        client: OpenAI client instance.
        max_retries: Maximum number of retry attempts.
        **kwargs: Arguments passed to client.chat.completions.create.

    Returns:
        The LLM response object.

    Raises:
        The last exception raised after all retries fail.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 60

            response = client.chat.completions.create(**kwargs)
            return response

        except (openai.APIError, openai.Timeout, openai.RateLimitError,
                openai.APIConnectionError) as e:
            last_exception = e

            if attempt < max_retries - 1:
                wait_time = min(4 * (2 ** attempt), 10)
                logger.warning(f"[LLM] Call failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.warning(f"[LLM] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"[LLM] All retries failed: {e}")

        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {e}")
            raise

    if last_exception:
        raise last_exception

    raise RuntimeError("LLM call failed for an unknown reason")

def _call_llm_with_prompt(prompt: str, client: Optional[OpenAI]) -> Dict[str, Any]:
    if client is None: return {}
    try:
        response = call_llm_with_retry(
            client,
            max_retries=3,
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        text = response.choices[0].message.content.strip()
        json_blob = extract_json_object(text)
        if not json_blob:
            logger.warning("[LLM] Response missing JSON")
            return {}
        return json.loads(json_blob)
    except Exception as e:
        logger.warning("[LLM] Call failed: %s", e)
        return {}

def call_llm_for_context(ctx: FunctionContext, pr_intent: Dict[str, Any], client: Optional[OpenAI]) -> List[Dict[str, Any]]:
    all_probes = []
    unit_used = 0
    integration_used = 0
    context_seen_ids: Set[str] = set()
    
    if ctx.unit_budget > 0 and not ctx.is_pure_new:
        prompt = build_code_diff_prompt(ctx, pr_intent)
        resp = _call_llm_with_prompt(prompt, client)
        probes = deduplicate_probes(ctx, resp, global_seen_ids=context_seen_ids, unit_cap=ctx.unit_budget, integration_cap=0)
        for p in probes: p["reason"] = "code_diff_coverage"
        all_probes.extend(probes)
        du, di = _count_probes_by_type(probes)
        unit_used += du
        integration_used += di
        
    remaining_unit = max(0, ctx.unit_budget - unit_used)
    remaining_int = max(0, ctx.integration_budget - integration_used)
    
    if remaining_unit > 0 or remaining_int > 0:
        prompt = build_llm_prompt(ctx, pr_intent, remaining_unit, remaining_int)
        resp = _call_llm_with_prompt(prompt, client)
        probes = deduplicate_probes(ctx, resp, global_seen_ids=context_seen_ids, unit_cap=remaining_unit, integration_cap=remaining_int)
        all_probes.extend(probes)
        
    return all_probes

def create_client(api_key: Optional[str], base_url: Optional[str]) -> Optional[OpenAI]:
    resolved_key = resolve_llm_api_key(api_key)
    # Configure this endpoint if you use a different OpenAI-compatible API provider.
    resolved_url = base_url or OPENAI_COMPATIBLE_BASE_URL
    return OpenAI(**build_openai_client_kwargs(api_key=resolved_key, base_url=resolved_url))

def generate_test_specs_pipeline(
    repo: str,
    pr_number: int,
    base_dir: str = DEFAULT_BASE_DIR,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    targets_view, targets_view_path = load_targets_view(base_dir, repo, pr_number)
    pr_intent = targets_view.get("pr_intent") or {}
    contexts = build_function_contexts(targets_view)
    
    select_integration_entries(contexts)
    assign_budgets(contexts)
    
    client = create_client(api_key, base_url)
    
    logger.info("Generating PROBES for %s PR #%s", repo, pr_number)
    logger.info("Target Functions: %d", len(contexts))
    
    final_probes = []
    
    for ctx in contexts:
        if ctx.unit_budget == 0 and ctx.integration_budget == 0:
            continue
            
        logger.info("  > Processing %s (Budget: U=%s, I=%s)", ctx.callable_name, ctx.unit_budget, ctx.integration_budget)
        probes = call_llm_for_context(ctx, pr_intent, client)
        
        for p in probes:
            p["module"] = ctx.module
            p["entry_kind"] = ctx.entry_kind
            
        final_probes.extend(probes)
        logger.info("    Generated %d probes", len(probes))

    output_dir = Path(base_dir) / "test_probe_generation" / "output" / repo
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pr_{pr_number}_test_specs.json"
    
    output_payload = {
        "pr_number": pr_number,
        "repo": repo,
        "probes": final_probes,
        "meta": {"type": "probe_based"}
    }
    
    with open(output_path, "w") as fh:
        json.dump(output_payload, fh, indent=2)
        
    logger.info("Saved %d probes to %s", len(final_probes), output_path)
    return str(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    
    try:
        generate_test_specs_pipeline(args.repo, args.pr, args.base_dir, args.api_key, args.base_url)
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
