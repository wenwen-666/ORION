#!/usr/bin/env python3
"""
PR Intent generator.
Builds PR-level intent JSON from a PR metadata file.

Input: PR metadata JSON for a single repository pull request.
Output: <ORION>/intent_extraction/output/{repo}/pr_{number}_intent.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LLM_MODEL, OPENAI_COMPATIBLE_BASE_URL, build_openai_client_kwargs, resolve_llm_api_key
from openai import OpenAI


class PRIntentGenerator:
    """Generate PR-level intent metadata."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the PR intent generator.

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

    def generate_pr_intent(self, pr_metadata_file: str, output_file: str = None, repo_name: str = None):
        """
        Generate PR intent from a PR metadata JSON file.

        Args:
            pr_metadata_file: Input PR metadata JSON path.
            output_file: Optional output intent JSON path.
            repo_name: Optional repository name.
        """
        print(f"\n{'='*60}")
        print("Generate PR Intent")
        print(f"{'='*60}\n")

        print("[1/4] Load PR metadata...")
        with open(pr_metadata_file, 'r', encoding='utf-8') as f:
            pr_data = json.load(f)

        pr_number = pr_data['pr_number']
        pr_title = pr_data.get('title', 'Unknown PR')
        pr_description = pr_data.get('description', None)
        diff = pr_data.get('diff', '')
        changed_files = pr_data.get('changed_files', [])

        if repo_name is None:
            input_path = Path(pr_metadata_file)
            repo_name = input_path.parent.name

        if output_file is None:
            output_dir = Path(__file__).parent / "output" / repo_name
            output_file = output_dir / f"pr_{pr_number}_intent.json"

        print(f"  ✓ PR #{pr_number}: {pr_title}")
        print(f"  ✓ Changed files: {len(changed_files)}")

        print("\n[2/4] Call the LLM to generate PR intent...")
        try:
            pr_intent = self._extract_pr_intent(pr_title, pr_description, diff, changed_files)
            print("  ✓ Intent generated successfully")
            print(f"    Change type: {pr_intent.get('change_type', 'N/A')}")
            print(f"    Expected behavior change: {pr_intent.get('expected_behavior_change', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Failed to generate intent: {e}")
            pr_intent = self._get_fallback_intent(pr_title)

        print("\n[3/4] Assemble output data...")
        output_data = {
            "pr_number": pr_number,
            "pr_title": pr_title,
            "pr_description": pr_description,
            "pr_url": pr_data.get('url', ''),
            "pre_commit": pr_data.get('pre_commit', ''),
            "post_commit": pr_data.get('post_commit', ''),
            "changed_files": changed_files,
            "intent": pr_intent
        }

        print("\n[4/4] Save results...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to: {output_file}")
        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}\n")

    def _extract_pr_intent(self, title: str, description: str, diff: str, changed_files: List[str]) -> Dict[str, Any]:
        """
        Extract PR intent using the LLM.

        Args:
            title: PR title.
            description: PR description.
            diff: Full diff text.
            changed_files: List of changed files.

        Returns:
            Parsed intent dictionary.
        """
        system_prompt = """You are a software engineering assistant specialized in analyzing Pull Requests.
Your task is to analyze a PR and extract structured intent information, including risky compatibility changes.

Analyze the PR and provide:
1. change_type: The type of change (optimization | bug_fix | refactor | feature | mixed | unknown)
2. expected_behavior_change: The scope of behavior change (none | local | global | unknown)
3. changed_functions: List of function names that were modified (extract from diff)
4. non_functional_goals: List of non-functional goals (e.g., performance_up, readability, maintainability)
5. high_level_expectations: List of high-level expectations for the change
6. backward_compatibility: Breaking-change assessment with explicit mitigation steps.

Output ONLY valid JSON, no extra text."""

        title = title or "Unknown PR"
        description_text = (description[:500] if description else "No description provided")
        diff = diff or ""
        changed_files = changed_files or []

        diff_truncated = diff[:5000] if len(diff) > 5000 else diff

        user_prompt = f"""**PR Information:**
- Title: {title}
- Description: {description_text}

**Changed Files:**
{', '.join(changed_files[:10])}

**Diff (truncated):**
```diff
{diff_truncated}
```

Output a JSON with exactly these fields:
{{
  "change_type": "optimization | bug_fix | refactor | feature | mixed | unknown",
  "expected_behavior_change": "none | local | global | unknown",
  "changed_functions": ["function1", "function2"],
  "non_functional_goals": ["performance_up", "readability"],
  "high_level_expectations": [
    "for normal inputs, behavior should stay the same",
    "performance should improve"
  ],
  "backward_compatibility": {{
    "is_breaking": true,
    "removed_features": ["removed argument names or APIs"],
    "migration_path": "Describe how users should update their code (e.g., use dump_default instead of default)."
  }}
}}

Important:
- Be specific and concise
- Focus on the WHAT and WHY, not the HOW
- Extract function names from the diff (look for def function_name patterns)
- List specific expectations that can be tested
- Explicitly mention removals/deprecations in backward_compatibility.removed_features and give a concrete migration path"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "X-Title": "PR Intent Extraction Tool",
                },
                model=self.model,
                messages=messages,
                max_tokens=1500
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        if not completion or not completion.choices:
            raise RuntimeError("LLM returned empty completion")

        response = completion.choices[0].message.content

        if not response:
            raise RuntimeError("LLM returned empty response content")

        return self._parse_intent_response(response)

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed intent dictionary.
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

            required_fields = [
                'change_type',
                'expected_behavior_change',
                'changed_functions',
                'non_functional_goals',
                'high_level_expectations',
                'backward_compatibility',
            ]
            for field in required_fields:
                if field not in result:
                    if field in ['changed_functions', 'non_functional_goals', 'high_level_expectations']:
                        result[field] = []
                    elif field == 'backward_compatibility':
                        result[field] = {}
                    else:
                        result[field] = "unknown"

            bc = result.get('backward_compatibility')
            if not isinstance(bc, dict):
                bc = {}
            if 'is_breaking' not in bc:
                bc['is_breaking'] = False
            if 'removed_features' not in bc or not isinstance(bc.get('removed_features'), list):
                bc['removed_features'] = []
            if 'migration_path' not in bc:
                bc['migration_path'] = ""
            result['backward_compatibility'] = bc

            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠ Failed to parse JSON: {e}")
            print(f"  Response preview: {response[:200]}...")
            raise

    def _get_fallback_intent(self, title: str) -> Dict[str, Any]:
        """Return a fallback intent when the LLM call fails."""
        return {
            "change_type": "unknown",
            "expected_behavior_change": "unknown",
            "changed_functions": [],
            "non_functional_goals": [],
            "high_level_expectations": [title],
             "backward_compatibility": {
                 "is_breaking": False,
                 "removed_features": [],
                 "migration_path": ""
             },
            "error": "LLM call failed"
        }


def main():
    """Command-line entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate PR intent from PR metadata'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to the PR metadata JSON file'
    )
    parser.add_argument(
        '--output',
        required=False,
        help='Optional output intent JSON path (default: output/{repo}/pr_{pr}_intent.json)'
    )
    parser.add_argument(
        '--repo',
        required=False,
        help='Optional repository name (otherwise inferred from the input path)'
    )

    args = parser.parse_args()

    generator = PRIntentGenerator()
    generator.generate_pr_intent(args.input, args.output, args.repo)


if __name__ == '__main__':
    main()
