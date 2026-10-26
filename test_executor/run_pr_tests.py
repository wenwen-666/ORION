#!/usr/bin/env python3
"""Run generated probe tests in isolated pre- and post-commit environments.

This stage prepares or reuses the matching conda environments, executes the
generated pytest runner for both commits, and stores raw execution results for
later summarization.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Tuple

# Import CondaEnvManager from test_executor directory
WORK_ROOT = Path(os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(WORK_ROOT / "test_executor"))

from conda_env_manager import CondaEnvManager


def run_pytest_in_env(
    env_name: str,
    test_file: Path,
    project_root: Path,
    output_dir: Path,
    manager: CondaEnvManager,
) -> int:
    """
    Run pytest in a given conda environment.

    Args:
        env_name: Name of the conda environment
        test_file: Path to the test file to execute
        project_root: Root directory of the project
        output_dir: Directory where test outputs should be written
        manager: CondaEnvManager instance

    Returns:
        Exit code from pytest
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running pytest in environment: {env_name}")
    print(f"  Test file: {test_file}")
    print(f"  Project root: {project_root}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*70}\n")

    # Build environment variables
    env_vars = os.environ.copy()

    # Set PR_TEST_OUTPUT_DIR for tests to write JSON outputs
    env_vars["PR_TEST_OUTPUT_DIR"] = str(output_dir)

    # Build PYTHONPATH: DO NOT include project_root to avoid importing from source checkout
    # Tests should import from conda env's site-packages, not from repo checkout
    pythonpath_parts = [
        str(WORK_ROOT),           # For test_pipeline tools if needed
        str(test_file.parent),    # For pytest to find tests file
    ]

    # Append existing PYTHONPATH if set
    if "PYTHONPATH" in env_vars:
        pythonpath_parts.append(env_vars["PYTHONPATH"])

    env_vars["PYTHONPATH"] = ":".join(pythonpath_parts)

    print(f"Environment variables:")
    print(f"  PR_TEST_OUTPUT_DIR={env_vars['PR_TEST_OUTPUT_DIR']}")
    print(f"  PYTHONPATH={env_vars['PYTHONPATH'][:100]}...")
    print()

    try:
        env_prefix = manager.get_env_prefix(env_name)
    except RuntimeError as exc:
        print(f"ERROR: Unable to locate conda environment {env_name}: {exc}")
        return 1

    # Use conda run to execute pytest in the specified environment
    cmd = [
        "conda", "run", "-p", env_prefix, "--no-capture-output",
        "pytest", str(test_file),
        "-v",
        "-s",
        "--tb=short",
    ]

    print(f"Executing command:")
    print(f"  {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(test_file.parent),  # Use test file directory, NOT project_root
            env=env_vars,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Print summary of output (first and last few lines)
        print(f"{'='*70}")
        print(f"Pytest execution completed with exit code: {result.returncode}")
        print(f"{'='*70}")

        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 20:
                print("\nStdout:")
                print(result.stdout)
            else:
                print(f"\nStdout (first 10 lines):")
                print('\n'.join(lines[:10]))
                print(f"\n... ({len(lines) - 20} lines omitted) ...\n")
                print(f"Stdout (last 10 lines):")
                print('\n'.join(lines[-10:]))

        if result.stderr:
            print("\nStderr:")
            stderr_lines = result.stderr.strip().split('\n')
            if len(stderr_lines) <= 10:
                print(result.stderr)
            else:
                print('\n'.join(stderr_lines[:10]))
                print(f"... ({len(stderr_lines) - 10} more lines)")

        print()

        return result.returncode

    except subprocess.TimeoutExpired:
        print(f"ERROR: Pytest execution timed out after 600 seconds")
        return -1
    except Exception as e:
        print(f"ERROR: Pytest execution failed with exception: {e}")
        return -1


def main() -> int:
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Run pytest on PR test cases in pre/post conda environments"
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="PR number"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository name (e.g., marshmallow, pandas)"
    )
    parser.add_argument(
        "--pre-commit",
        type=str,
        required=True,
        help="Commit hash before the PR"
    )
    parser.add_argument(
        "--post-commit",
        type=str,
        required=True,
        help="Commit hash after/head of the PR"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Absolute path to the repo checkout"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(WORK_ROOT),
        help="Base directory for test files"
    )

    args = parser.parse_args()

    pr_number = args.pr_number
    repo = args.repo
    pre_commit = args.pre_commit
    post_commit = args.post_commit
    project_root = Path(args.project_root)
    base_dir = Path(args.base_dir)

    print(f"\n{'='*70}")
    print(f"PR Test Executor")
    print(f"{'='*70}")
    print(f"PR Number: {pr_number}")
    print(f"Repository: {repo}")
    print(f"Pre-commit: {pre_commit}")
    print(f"Post-commit: {post_commit}")
    print(f"Project root: {project_root}")
    print(f"Base dir: {base_dir}")
    print(f"{'='*70}\n")

    # Validate project root exists
    if not project_root.exists():
        print(f"ERROR: Project root does not exist: {project_root}")
        return 1

    # Resolve path to test file (prefer probe runner output)
    probe_test_file = base_dir / "test_probe_generation" / "output" / repo / f"pr_{pr_number}_tests.py"
    legacy_test_file = base_dir / "test_case_generation" / "output" / repo / f"pr_{pr_number}_tests.py"

    if probe_test_file.exists():
        test_file = probe_test_file
    elif legacy_test_file.exists():
        test_file = legacy_test_file
    else:
        print(f"ERROR: Test file does not exist: {probe_test_file}")
        print(f"       (Legacy path also missing: {legacy_test_file})")
        print(f"\nPlease ensure the probe generation pipeline has completed for PR {pr_number}")
        return 1

    print(f"Test file located: {test_file}\n")

    # Instantiate CondaEnvManager
    manager = CondaEnvManager(base_dir=str(base_dir))

    # Create/reuse conda environments for pre/post commits
    print(f"Setting up conda environments...")
    try:
        pre_env, post_env = manager.setup_pr_environments(
            project=repo,
            pr_number=pr_number,
            pre_commit=pre_commit,
            post_commit=post_commit,
        )
        print(f"\nEnvironments ready:")
        print(f"  Pre-commit env: {pre_env}")
        print(f"  Post-commit env: {post_env}")
    except Exception as e:
        print(f"\nERROR: Failed to setup conda environments: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compute output directories
    pre_output_dir = WORK_ROOT / "test_outputs" / repo / f"pr_{pr_number}" / "pre"
    post_output_dir = WORK_ROOT / "test_outputs" / repo / f"pr_{pr_number}" / "post"

    # Run pytest in pre environment
    print(f"\n{'#'*70}")
    print(f"# PHASE 1: Running tests in PRE-commit environment")
    print(f"{'#'*70}")
    pre_code = run_pytest_in_env(pre_env, test_file, project_root, pre_output_dir, manager)

    # Run pytest in post environment
    print(f"\n{'#'*70}")
    print(f"# PHASE 2: Running tests in POST-commit environment")
    print(f"{'#'*70}")
    post_code = run_pytest_in_env(post_env, test_file, project_root, post_output_dir, manager)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - PR {pr_number}")
    print(f"{'='*70}")
    print(f"[PR {pr_number}] pre run  (env={pre_env}):  exit code {pre_code}, outputs at {pre_output_dir}")
    print(f"[PR {pr_number}] post run (env={post_env}): exit code {post_code}, outputs at {post_output_dir}")
    print(f"{'='*70}\n")

    # Count output files
    try:
        pre_count = len(list(pre_output_dir.glob("*.json"))) if pre_output_dir.exists() else 0
        post_count = len(list(post_output_dir.glob("*.json"))) if post_output_dir.exists() else 0
        print(f"Output files generated:")
        print(f"  Pre:  {pre_count} JSON files")
        print(f"  Post: {post_count} JSON files")
    except Exception:
        pass

    print(f"\nTest execution completed successfully.")
    print(f"Note: Non-zero exit codes are expected and will be analyzed in the summarization step.\n")

    # Always return 0 - behavior differences captured in JSON outputs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
