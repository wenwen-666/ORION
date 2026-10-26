#!/usr/bin/env python3
"""
Function-level oracle generator.

For each changed function, generate an initial partition of
must_preserve and may_change regions.

Input: changed-function code from impact_region.json plus pr_intent.json.
Output: function-level oracle definitions.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LLM_MODEL, OPENAI_COMPATIBLE_BASE_URL, build_openai_client_kwargs, resolve_llm_api_key
from openai import OpenAI

ORACLE_MAX_FUNCS_PER_PR = 16


class FunctionOracleGenerator:
    """Generate function-level oracles."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the function-oracle generator.

        Args:
            api_key: API key for the configured LLM provider.
            base_url: API base URL.
            model: Model name.
        """
        self.api_key = resolve_llm_api_key(api_key)
        # Configure this endpoint if you use a different OpenAI-compatible API provider.
        self.base_url = base_url or OPENAI_COMPATIBLE_BASE_URL
        self.model = model

        self.client = OpenAI(**build_openai_client_kwargs(api_key=self.api_key, base_url=self.base_url))

    def _compute_oracle_priority(
        self,
        func: Dict[str, Any],
        pr_intent: Dict[str, Any],
        call_graph: Dict[str, Any],
    ) -> int:
        """
        Cheap static priority for oracle pre-filtering.
        Higher score = more likely worth spending oracle budget.
        """
        function_id = func.get("function_id") or func.get("qualified_name", "")
        function_name = func.get("name", function_id.split(".")[-1] if function_id else "")

        intent = pr_intent.get("intent", {}) or {}
        intent_changed = intent.get("changed_functions", []) or []

        is_target = any(function_name in cf for cf in intent_changed)
        score = 100 if is_target else 0

        out_degree = 0
        if function_id and function_id in (call_graph or {}):
            out_degree = len(call_graph.get(function_id, []) or [])
        score += min(out_degree, 10) * 3

        in_degree = 0
        if function_name and call_graph:
            for _, edges in call_graph.items():
                for e in edges or []:
                    callee = e.get("callee")
                    if callee == function_name:
                        in_degree += 1
        score += min(in_degree, 10) * 2

        src_after = func.get("source_after") or ""
        score += min(len(src_after.splitlines()), 80) // 10

        callability = func.get("callability") or {}
        entry_kind = callability.get("entry_kind", "function")
        if entry_kind in ("constructor", "method"):
            score += 2

        control_flow = func.get("control_flow_metrics") or {}
        branch_count = control_flow.get("branch_count", 0)
        loop_count = control_flow.get("loop_count", 0)
        try_count = control_flow.get("try_count", 0)
        match_count = control_flow.get("pattern_match_count", 0)
        ternary_count = control_flow.get("conditional_expression_count", 0)
        complexity_score = control_flow.get("complexity_score", 0)

        score += min(branch_count, 12) * 2
        score += min(loop_count, 6) * 3
        score += min(try_count, 4) * 4
        score += min(match_count, 3) * 4
        score += min(ternary_count, 4) * 2
        score += min(complexity_score, 30)

        if control_flow.get("has_control_flow"):
            score += 5

        return score

    def _get_skipped_oracle(self, func: Dict[str, Any], reason: str = "low_priority_over_cap") -> Dict[str, Any]:
        """
        Oracle stub for skipped functions.
        MUST NOT set 'error', so downstream can treat it as intentionally skipped.
        """
        fid = func.get('function_id') or func.get('qualified_name', 'unknown')
        return {
            "function_id": fid,
            "code_diff_summary": [],
            "setup_requirements": [],
            "must_preserve_region": [],
            "may_change_region": [],
            "behavior_notes": [f"oracle_skipped:{reason}"],
            "control_flow_hints": {
                "new_paths": [],
                "important_conditions": []
            },
            "skipped": reason
        }

    def generate_oracles(self, impact_region_file: str, pr_intent_file: str, output_file: str = None, repo_name: str = None):
        """
        Generate function-level oracles from impact_region and pr_intent.

        Args:
            impact_region_file: Path to impact_region.json.
            pr_intent_file: Path to pr_intent.json.
            output_file: Optional oracle output path.
            repo_name: Optional repository name.
        """
        print(f"\n{'='*60}")
        print("Generate function-level oracles")
        print(f"{'='*60}\n")

        print("[1/4] Load impact_region...")
        with open(impact_region_file, 'r', encoding='utf-8') as f:
            impact_data = json.load(f)

        print("[2/4] Load pr_intent...")
        with open(pr_intent_file, 'r', encoding='utf-8') as f:
            pr_intent = json.load(f)

        pr_number = pr_intent.get('pr_number', 'unknown')

        if repo_name is None:
            input_path = Path(impact_region_file)
            repo_name = input_path.parent.name

        if output_file is None:
            output_dir = Path(__file__).parent / "output" / repo_name
            output_file = output_dir / f"pr_{pr_number}_function_oracles.json"

        print(f"  ✓ PR #{pr_intent.get('pr_number', 'N/A')}: {pr_intent.get('pr_title', 'N/A')}")

        if 'changed_functions' in impact_data:
            raw_changed_functions = impact_data.get('changed_functions', [])
        else:
            raw_changed_functions = [
                func for func in impact_data.get('functions', [])
                if func.get('role') == 'changed'
            ]

        changed_functions = []
        test_functions = []
        for func in raw_changed_functions:
            kind = func.get('kind')
            if kind is None or kind == 'code':
                changed_functions.append(func)
            elif kind == 'test':
                test_functions.append(func)

        print(f"  ✓ Found {len(changed_functions)} changed production functions and {len(test_functions)} changed test functions for reference")
        test_lookup = {
            func.get('function_id'): func
            for func in test_functions
            if func.get('function_id')
        }

        print("\n[3/4] Generate an oracle for each function...")
        oracles = []

        call_graph = impact_data.get("call_graph", {}) or {}

        # ---- NEW: PR-level oracle pre-filtering (Top-K) ----
        scored = []
        for func in changed_functions:
            score = self._compute_oracle_priority(func, pr_intent, call_graph)
            scored.append((score, func))

        scored.sort(key=lambda x: -x[0])

        top_funcs = [f for _, f in scored[:ORACLE_MAX_FUNCS_PER_PR]]
        rest_funcs = [f for _, f in scored[ORACLE_MAX_FUNCS_PER_PR:]]

        print(f"  ✓ Oracle budget: top-{ORACLE_MAX_FUNCS_PER_PR} / {len(changed_functions)}")
        if rest_funcs:
            print(f"  ✓ Skip LLM oracle generation for {len(rest_funcs)} lower-priority functions")

        for i, func in enumerate(top_funcs, 1):
            print(f"\n  [{i}/{len(top_funcs)}] Process function: {func.get('function_id', 'unknown')}")

            try:
                reference_usages = self._collect_reference_usages(func, call_graph, test_lookup)
                oracle = self._generate_function_oracle(func, pr_intent, reference_usages)
                print("    ✓ Oracle generated successfully")
                print(f"      code_diff_summary: {len(oracle.get('code_diff_summary', []))} items")
                print(f"      must_preserve: {len(oracle.get('must_preserve_region', []))} items")
                print(f"      may_change: {len(oracle.get('may_change_region', []))} items")
                control_flow = oracle.get('control_flow_hints', {})
                print(f"      control_flow: {len(control_flow.get('new_paths', []))} paths, {len(control_flow.get('important_conditions', []))} conditions")
                oracles.append(oracle)
            except Exception as e:
                print(f"    ✗ Oracle generation failed: {e}")
                oracles.append(self._get_fallback_oracle(func))

        for func in rest_funcs:
            oracles.append(self._get_skipped_oracle(func))

        print("\n[4/4] Save results...")
        output_data = {
            "pr_number": pr_intent.get('pr_number'),
            "pr_title": pr_intent.get('pr_title'),
            "function_oracles": oracles
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to: {output_file}")
        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}\n")

    def _collect_reference_usages(
        self,
        func: Dict[str, Any],
        call_graph: Dict[str, Any],
        test_lookup: Dict[str, Dict[str, Any]],
        limit: int = 3,
    ) -> List[str]:
        """Collect snippets from changed tests that invoke the target function."""
        func_name = func.get("name")
        if not func_name or not call_graph or not test_lookup:
            return []

        usages: List[str] = []
        for caller_id, edges in call_graph.items():
            test_entry = test_lookup.get(caller_id)
            if not test_entry:
                continue
            matched_lines = [edge.get("line") for edge in (edges or []) if edge.get("callee") == func_name]
            if not matched_lines:
                continue
            snippet = (test_entry.get("source_after") or test_entry.get("source_before") or "").strip()
            if not snippet:
                continue
            header = f"# {caller_id}"
            line_nums = ", ".join(str(line) for line in matched_lines if line)
            if line_nums:
                header += f" (lines {line_nums})"
            usages.append(f"{header}\n{snippet[:1200]}")
            if len(usages) >= limit:
                break
        return usages

    def _generate_function_oracle(self, func: Dict[str, Any], pr_intent: Dict[str, Any], reference_usages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate the full oracle payload for a single function.

        Args:
            func: Function metadata including before/after source.
            pr_intent: PR intent payload.

        Returns:
            Parsed oracle dictionary.
        """
        system_prompt = """You are a senior software engineer specialized in test oracle design and code change analysis.
Your task for each function modification is to:
1) Summarize the key BEHAVIORAL changes introduced by this patch.
2) Partition the input space into:
   - must_preserve_region: scenarios where behavior MUST remain the same after this PR
   - may_change_region: scenarios where behavior MAY legitimately change (bug fixes, stricter validation, new formats, etc.)
3) Identify important new control-flow paths and conditions.
4) Derive concrete setup requirements that describe how to instantiate objects / configure environment based on the provided Reference Usage.

Output ONLY valid JSON, no extra text."""

        source_before = func.get('source_before', 'N/A')
        source_after = func.get('source_after', 'N/A')

        function_id = func.get('function_id') or func.get('qualified_name', 'unknown')
        function_name = func.get('name', function_id.split('.')[-1] if '.' in function_id else function_id)

        intent = pr_intent.get('intent', {})
        change_type = intent.get('change_type', 'unknown')
        expected_behavior_change = intent.get('expected_behavior_change', 'unknown')
        changed_functions = intent.get('changed_functions', [])
        non_functional_goals = intent.get('non_functional_goals', [])
        high_level_expectations = intent.get('high_level_expectations', [])
        compat = intent.get('backward_compatibility', {}) or {}

        static_analysis = func.get("static_analysis") or {}
        reference_usages = reference_usages or []
        reference_usage_block = "\n\n".join(reference_usages[:3]) if reference_usages else "No test usage found in diff."
        warnings: List[str] = []

        if static_analysis.get("risk_silent_swallow"):
            removed = static_analysis.get("removed_args", []) or []
            removed_str = ", ".join(removed) if removed else "removed arguments"
            warnings.append(
                f"""
[🚨 CRITICAL REGRESSION RISK: Silent Parameter Swallow]
- DETECTED: Arguments [{removed_str}] were removed, but `**kwargs` remains.
- RISK: Passing [{removed_str}] will NOT raise TypeError; they will be silently ignored.
- REQUIREMENT: Add a must_preserve_region rule verifying that passing [{removed_str}] explicitly raises a clear error (TypeError / ValidationError) and is NOT silently swallowed.
"""
            )

        if static_analysis.get("changed_defaults"):
            for default_change in static_analysis.get("changed_defaults", []):
                warnings.append(
                    f"""
[⚠️ WARNING: Default Value Changed]
- DETECTED: Parameter '{default_change['arg']}' default changed from `{default_change['old']}` to `{default_change['new']}`.
- REQUIREMENT: Add (1) a may_change_region describing this intentional default change, and (2) a must_preserve_region verifying the function behaves correctly when the argument is omitted (new default path).
"""
                )

        if static_analysis.get("risk_exception_drift"):
            rem_list = static_analysis.get("exception_changes", {}).get("removed") or []
            add_list = static_analysis.get("exception_changes", {}).get("added") or []
            rem_str = ", ".join(rem_list) if rem_list else "previous exceptions"
            add_str = ", ".join(add_list) if add_list else "new exceptions"
            warnings.append(
                f"""
[⚠️ WARNING: Exception Contract Changed]
- DETECTED: Function stopped raising [{rem_str}] and now raises [{add_str}].
- RISK: Downstream try/except handlers for [{rem_str}] will no longer catch the error, causing uncaught exceptions.
- REQUIREMENT: Add a must_preserve_region to confirm whether this exception change is intentional or a regression.
"""
            )

        if static_analysis.get("risk_generator_change"):
            warnings.append(
                """
[⚠️ WARNING: Return Type Narrowing (Generator)]
- DETECTED: Function now yields a generator/lazy iterator instead of returning a concrete collection.
- RISK: Downstream consumers expecting len(), indexing, or repeated iteration will break.
- REQUIREMENT: Add a must_preserve_region verifying whether collection semantics (len/indexing/multiple iteration) are still supported or documenting the breaking change.
"""
            )

        security_context = "\n".join(warnings).strip()

        is_target_function = any(function_name in cf for cf in changed_functions)

        callability = func.get("callability") or {}
        entry_kind = callability.get("entry_kind", "function")
        class_qualname = callability.get("class_qualname") or "N/A"

        user_prompt = f"""You are analyzing a single function inside a PR.

[Function Info]
- function_id: {function_id}
- file: {func.get('file', 'unknown')}
- is_target_function_in_pr: {is_target_function}

[Callability]
- entry_kind: {entry_kind}
- class_qualname: {class_qualname}

[PR intent (global)]
- change_type: {change_type}
- expected_behavior_change: {expected_behavior_change}
- non_functional_goals: {', '.join(non_functional_goals) or '[]'}
- high_level_expectations:
{chr(10).join(f'  - {exp}' for exp in high_level_expectations)}

[Backward Compatibility Intent]
- is_breaking: {compat.get('is_breaking', False)}
- removed_features: {', '.join(compat.get('removed_features', [])) or '[]'}
- migration_path: {compat.get('migration_path', '') or 'N/A'}

[Reference Usage (from changed tests)]
```python
{reference_usage_block}
```

[Static Code Analysis]
{json.dumps(static_analysis, indent=2)}
{security_context if security_context else ''}

[Function source BEFORE]
```python
{source_before[:2000]}
```

[Function source AFTER]
```python
{source_after[:2000]}
```

Based on the PR intent and the concrete code changes, output a JSON object with EXACTLY the following fields:

{{
  "function_id": "{function_id}",
  "code_diff_summary": [
    "... short English sentences summarizing the most important BEHAVIORAL changes (1-5 items) ..."
  ],
  "setup_requirements": [
    "... instructions for instantiating classes, configuring dependencies, or preparing data before calling the function ..."
  ],
  "must_preserve_region": [
    "... input scenarios where behavior MUST remain the same after this PR ..."
  ],
  "may_change_region": [
    "... input scenarios where behavior MAY legitimately change (bug fix, stricter validation, new format, performance, etc.) ..."
  ],
  "behavior_notes": [
    "... any important additional observations about expected behavior ..."
  ],
  "control_flow_hints": {{
    "new_paths": [
      "... describe new or modified control-flow paths, such as new branches or exception paths ..."
    ],
    "important_conditions": [
      "... important conditions that govern behavior, e.g. 'value < 0', 'raises OverflowError', etc. ..."
    ]
  }}
}}

Guidelines:
- Focus on WHAT kinds of inputs/behaviors, not HOW the code is implemented.
- setup_requirements must capture concrete initialization details derived from the Reference Usage (e.g., constructor arguments, fixtures, environment variables).
- must_preserve_region: typical correct-use scenarios that should continue working exactly as before.
- may_change_region: edge cases, invalid inputs, stricter validation, new supported formats, changed error types/messages, etc.
- code_diff_summary: 1-5 short sentences capturing the key observable behavior differences.
- control_flow_hints: only include the most important new/changed branches and conditions, not every tiny detail.
- When the function is a constructor or method, describe behavior in terms of how users construct and use the class, not by directly calling __init__ or unbound methods.
- If any warnings are shown above, you MUST address each of them explicitly with appropriate setup_requirements and must_preserve_region / may_change_region coverage items."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion = self.client.chat.completions.create(
            extra_headers={
                "X-Title": "Function Oracle Generation Tool",
            },
            model=self.model,
            messages=messages,
            max_tokens=2000
        )

        response = completion.choices[0].message.content

        return self._parse_oracle_response(response, func)

    def _normalize_str_list(self, value) -> List[str]:
        """
        Normalize a list-like value into a clean list of strings.
        """
        if not value:
            return []
        out = []
        for x in value:
            if isinstance(x, str):
                s = x.strip()
            else:
                s = str(x).strip()
            if s:
                out.append(s)
        return out

    def _parse_oracle_response(self, response: str, func: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response for a single oracle.

        Args:
            response: Raw LLM response text.
            func: Function metadata.

        Returns:
            Parsed oracle dictionary.
        """
        try:
            response = response.strip()

            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()

            result = json.loads(response)

            if 'function_id' not in result:
                result['function_id'] = func.get('function_id') or func.get('qualified_name', 'unknown')

            result['code_diff_summary'] = self._normalize_str_list(result.get('code_diff_summary'))
            result['setup_requirements'] = self._normalize_str_list(result.get('setup_requirements'))
            result['must_preserve_region'] = self._normalize_str_list(result.get('must_preserve_region'))
            result['may_change_region'] = self._normalize_str_list(result.get('may_change_region'))
            result['behavior_notes'] = self._normalize_str_list(result.get('behavior_notes'))

            cf = result.get('control_flow_hints', {})
            if not isinstance(cf, dict):
                cf = {}
            cf['new_paths'] = self._normalize_str_list(cf.get('new_paths'))
            cf['important_conditions'] = self._normalize_str_list(cf.get('important_conditions'))
            result['control_flow_hints'] = cf

            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠ Failed to parse JSON: {e}")
            print(f"  Response preview: {response[:200]}...")
            raise

    def _get_fallback_oracle(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return an empty fallback oracle when the LLM call fails.
        """
        fid = func.get('function_id') or func.get('qualified_name', 'unknown')
        return {
            "function_id": fid,
            "code_diff_summary": [],
            "setup_requirements": [],
            "must_preserve_region": [],
            "may_change_region": [],
            "behavior_notes": ["oracle_generation_failed"],
            "control_flow_hints": {
                "new_paths": [],
                "important_conditions": []
            },
            "error": "LLM call failed"
        }


def main():
    """Command-line entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate oracle definitions for changed functions'
    )
    parser.add_argument(
        '--impact-region',
        required=True,
        help='Path to impact_region.json'
    )
    parser.add_argument(
        '--pr-intent',
        required=True,
        help='Path to pr_intent.json'
    )
    parser.add_argument(
        '--output',
        required=False,
        help='Optional oracle output JSON path'
    )
    parser.add_argument(
        '--repo',
        required=False,
        help='Optional repository name (otherwise inferred from the input path)'
    )

    args = parser.parse_args()

    generator = FunctionOracleGenerator()
    generator.generate_oracles(args.impact_region, args.pr_intent, args.output, args.repo)


if __name__ == '__main__':
    main()
