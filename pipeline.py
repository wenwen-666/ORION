#!/usr/bin/env python3
"""
Unified entrypoint for the full regression-analysis pipeline.

Features:
1. Run a specific repo/PR or process all available tasks.
2. Include Impact Region generation as Stage 0.
3. Write logs to <ORION>/logs/<stage>/<repo>/...
4. Support resume behavior by skipping completed stages.
5. Record logs for all LLM-driven stages.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import (
    metadata_dir_for_type,
    output_base_for_type,
    primary_repo_metadata_dir,
    resolve_pr_metadata,
    resolve_project_base,
    resolve_project_checkout,
)


LOCAL_BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("ORION_BASE_DIR", str(LOCAL_BASE_DIR)))
LOG_BASE_DIR = BASE_DIR / "logs"
PROJECT_BASE = resolve_project_base(BASE_DIR)


def setup_logger(log_file: Path) -> logging.Logger:
    """Create a logger for a stage log file."""
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_command_with_log(cmd: List[str], log_file: Path, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a command and capture its output in the stage log."""
    logger = setup_logger(log_file)

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"CWD: {cwd or os.getcwd()}")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 70)

    try:
        run_env = os.environ.copy()
        run_env.setdefault("ORION_BASE_DIR", str(BASE_DIR))
        run_env.setdefault("ORION_PROJECT_BASE", str(PROJECT_BASE))

        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=3600
        )

        logger.info("STDOUT:")
        logger.info(result.stdout)

        if result.stderr:
            logger.info("STDERR:")
            logger.info(result.stderr)

        logger.info(f"Exit code: {result.returncode}")

        return result.returncode == 0, result.stdout

    except subprocess.TimeoutExpired:
        logger.error("Command timeout")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Exception: {e}")
        return False, str(e)



class PRPipeline:
    """Run the full pipeline for a single PR."""

    def __init__(self, repo: str, pr: int, force: bool = False):
        self.repo = repo
        self.pr = pr
        self.force = force

        self.pr_metadata_path, self.pr_type = resolve_pr_metadata(BASE_DIR, repo, pr)
        self.output_base = output_base_for_type(BASE_DIR, self.pr_type)
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.pr_metadata = None
        self.pre_commit = None
        self.post_commit = None
        self.project_root = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log_dir = LOG_BASE_DIR / "pipeline" / repo / f"pr_{pr}"
        self.main_log_dir.mkdir(parents=True, exist_ok=True)
        self.main_log_file = self.main_log_dir / f"run_{timestamp}.log"

        self.logger = self._setup_main_logger()

    def _setup_main_logger(self) -> logging.Logger:
        """Configure the main pipeline logger."""
        logger = logging.getLogger(f"pipeline_{self.repo}_{self.pr}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console)

        file_handler = logging.FileHandler(self.main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

        return logger

    def validate(self) -> bool:
        """Validate the metadata and repository prerequisites for this PR."""
        self.logger.info("=" * 70)
        self.logger.info(f"Validating {self.repo} PR #{self.pr}")
        self.logger.info("=" * 70)

        if not self.pr_metadata_path.exists():
            self.logger.error(f"✗ PR metadata not found: {self.pr_metadata_path}")
            return False

        try:
            with open(self.pr_metadata_path, 'r') as f:
                self.pr_metadata = json.load(f)

            self.pre_commit = self.pr_metadata.get('pre_commit')
            self.post_commit = self.pr_metadata.get('post_commit')

            if not self.pre_commit or not self.post_commit:
                self.logger.error("✗ Missing commit information")
                return False

            self.logger.info(f"✓ Title: {self.pr_metadata.get('title', 'N/A')[:60]}")
            self.logger.info(f"✓ Pre: {self.pre_commit[:8]}...")
            self.logger.info(f"✓ Post: {self.post_commit[:8]}...")

        except Exception as e:
            self.logger.error(f"✗ Failed to load PR metadata: {e}")
            return False

        self.project_root = str(resolve_project_checkout(PROJECT_BASE, self.repo))

        if not Path(self.project_root).exists():
            self.logger.error(f"✗ Project root does not exist: {self.project_root}")
            return False

        self.logger.info(f"✓ Project: {self.project_root}")
        self.logger.info("")
        return True

    def should_run_stage(self, stage_num: int, stage_name: str) -> bool:
        """Decide whether a stage should run."""
        if self.force:
            return True

        if self.check_stage_completed(stage_num):
            self.logger.info(f"⏭️  Stage {stage_num} ({stage_name}) already complete, skipping")
            return False

        return True

    def check_stage_completed(self, stage_num: int) -> bool:
        """Check whether a stage has already finished."""
        if stage_num == 7:
            base = self.output_base / "output" / self.repo
            locations = [None, "all_results", "intended", "unintended"]
            suffixes = [""]
            for location in locations:
                search_dir = base / location if location else base
                if not search_dir.exists():
                    continue
                for suffix in suffixes:
                    patterns = [
                        f"pr_{self.pr}_regression_output{suffix}.json",
                        f"pr_{self.pr}_intendedness_output{suffix}.json",
                    ]
                    if any((search_dir / pattern).exists() for pattern in patterns):
                        return True
            return False

        checks = {
            0: BASE_DIR / "impact_region" / "output" / self.repo / f"pr_{self.pr}_impact.json",
            1: BASE_DIR / "intent_extraction" / "output" / self.repo / f"pr_{self.pr}_intent.json",
            2: BASE_DIR / "function_oracle_generation" / "output" / self.repo / f"pr_{self.pr}_function_oracles.json",
            3: BASE_DIR / "target_function_view" / "output" / self.repo / f"pr_{self.pr}_targets_view.json",
            4: BASE_DIR / "test_probe_generation" / "output" / self.repo / f"pr_{self.pr}_test_specs.json",
            5: BASE_DIR / "test_probe_generation" / "output" / self.repo / f"pr_{self.pr}_tests.py",
            6: BASE_DIR / "test_executor" / "output" / self.repo / f"pr_{self.pr}_behavior.json",
        }

        check_file = checks.get(stage_num)
        if check_file and check_file.exists():
            return True
        return False

    def get_stage_log(self, stage_num: int, stage_name: str) -> Path:
        """Return the log-file path for a pipeline stage."""
        stage_dir = LOG_BASE_DIR / f"stage{stage_num}_{stage_name.lower().replace(' ', '_')}" / self.repo
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"pr_{self.pr}.log"

    def get_output_path(self, *parts) -> Path:
        """Return an output path relative to the current PR output root."""
        return self.output_base / Path(*parts)

    def run_stage_0(self) -> bool:
        """Stage 0: build the impact region if it is missing."""
        if not self.should_run_stage(0, "impact_region"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 0: Impact Region")
        self.logger.info("=" * 70)

        impact_script = BASE_DIR / "impact_region" / "build_impact_region.py"
        if not impact_script.exists():
            self.logger.error(f"✗ Impact region script not found: {impact_script}")
            return False

        pr_metadata_dir = metadata_dir_for_type(BASE_DIR, self.pr_type)

        log_file = self.get_stage_log(0, "impact_region")
        cmd = [
            "python3", str(impact_script),
            "--repo", self.repo,
            "--pr", str(self.pr),
            "--base-dir", str(BASE_DIR),
            "--metadata-dir", str(pr_metadata_dir),
            "--repo-path", str(self.project_root),
            "--clones-base", str(PROJECT_BASE)
        ]

        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"  Log: {log_file}")
        self.logger.info(f"  {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_1(self) -> bool:
        """Stage 1: Intent Extraction (LLM)"""
        if not self.should_run_stage(1, "intent_extraction"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 1: Intent Extraction (LLM)")
        self.logger.info("=" * 70)

        log_file = self.get_stage_log(1, "intent_extraction")
        cmd = [
            "python3",
            str(BASE_DIR / "intent_extraction" / "generate_pr_intent.py"),
            "--input", str(self.pr_metadata_path),
            "--repo", self.repo
        ]

        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"  Log: {log_file}")
        self.logger.info(f"  {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_2(self) -> bool:
        """Stage 2: Function Oracle Generation (LLM)"""
        if not self.should_run_stage(2, "oracle_generation"):
            return True

        impact_file = BASE_DIR / "impact_region" / "output" / self.repo / f"pr_{self.pr}_impact.json"
        if not impact_file.exists():
            self.logger.error(f"✗ Impact region output not found: {impact_file}")
            self.logger.error("  Run Stage 0 first or generate the impact region manually")
            return False

        self.logger.info("=" * 70)
        self.logger.info("Stage 2: Function Oracle Generation (LLM)")
        self.logger.info("=" * 70)

        intent_file = BASE_DIR / "intent_extraction" / "output" / self.repo / f"pr_{self.pr}_intent.json"
        log_file = self.get_stage_log(2, "oracle_generation")

        cmd = [
            "python3",
            str(BASE_DIR / "function_oracle_generation" / "generate_function_oracle.py"),
            "--impact-region", str(impact_file),
            "--pr-intent", str(intent_file),
            "--repo", self.repo
        ]

        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"  Log: {log_file}")
        self.logger.info(f"  {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_3(self) -> bool:
        """Stage 3: Target Function View"""
        if not self.should_run_stage(3, "target_view"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 3: Target Function View")
        self.logger.info("=" * 70)

        log_file = self.get_stage_log(3, "target_view")
        cmd = [
            "python3",
            str(BASE_DIR / "target_function_view" / "build_target_view.py"),
            str(BASE_DIR), self.repo, str(self.pr)
        ]

        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"  Log: {log_file}")
        self.logger.info(f"  {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_4(self) -> bool:
        """Stage 4: Probe Generation (LLM + Coverage + Augment)"""
        if not self.should_run_stage(4, "probe_generation"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 4: Probe Generation (LLM + Coverage + Augment)")
        self.logger.info("=" * 70)

        # 4a: Base Probe Generation (LLM)
        self.logger.info("  4a: Base Probe Generation (LLM)")
        log_file = self.get_stage_log(4, "test_specs_base")
        cmd = [
            "python3",
            str(BASE_DIR / "test_probe_generation" / "probe_generation.py"),
            "--repo", self.repo, "--pr", str(self.pr),
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        if not success:
            self.logger.error("      ✗ Failed")
            return False
        self.logger.info("      ✓ Completed")

        # 4b: Oracle Coverage Check
        self.logger.info("  4b: Oracle Coverage Check")
        log_file = self.get_stage_log(4, "oracle_coverage")
        cmd = [
            "python3",
            str(BASE_DIR / "test_probe_generation" / "check_oracle_coverage.py"),
            "--repo", self.repo, "--pr", str(self.pr),
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        if not success:
            self.logger.warning("      ✗ Failed (treating as warning)")
            # Don't return False, allow pipeline to continue
        else:
            self.logger.info("      ✓ Completed")

        # 4c: Oracle-driven Augmentation (LLM)
        self.logger.info("  4c: Oracle-driven Probe Augmentation (LLM)")
        log_file = self.get_stage_log(4, "oracle_augment")
        cmd = [
            "python3",
            str(BASE_DIR / "test_probe_generation" / "augment_probe_from_oracle.py"),
            "--repo", self.repo, "--pr", str(self.pr),
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        self.logger.info(f"      {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_5(self) -> bool:
        """Stage 5: Probe Runner Generation"""
        if not self.should_run_stage(5, "probe_runner"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 5: Probe Runner Generation")
        self.logger.info("=" * 70)

        log_file = self.get_stage_log(5, "probe_runner")
        cmd = [
            "python3",
            str(BASE_DIR / "test_probe_generation" / "generate_probe_runner.py"),
            "--repo", self.repo,
            "--pr", str(self.pr),
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        self.logger.info(f"      {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_6(self) -> bool:
        """Stage 6: Test Execution (2 substages)"""
        if not self.should_run_stage(6, "test_execution"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 6: Test Execution")
        self.logger.info("=" * 70)

        # 6a: Run Tests
        self.logger.info("  6a: Run Tests")
        log_file = self.get_stage_log(6, "run_tests")

        cmd = [
            "python3", str(BASE_DIR / "test_executor" / "run_pr_tests.py"),
            "--repo", self.repo, "--pr-number", str(self.pr),
            "--pre-commit", self.pre_commit, "--post-commit", self.post_commit,
            "--project-root", self.project_root,
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        if not success:
            self.logger.error("      ✗ Failed")
            return False
        self.logger.info("      ✓ Completed")

        # 6b: Summarize Behavior
        self.logger.info("  6b: Summarize Behavior")
        log_file = self.get_stage_log(6, "summarize_behavior")
        cmd = [
            "python3", str(BASE_DIR / "test_executor" / "summarize_behavior.py"),
            "--repo", self.repo, "--pr-number", str(self.pr),
            "--base-dir", str(BASE_DIR)
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        self.logger.info(f"      {'✓ Completed' if success else '✗ Failed'}")
        self.logger.info("")
        return success

    def run_stage_7(self) -> bool:
        """Stage 7: Intendedness Classification."""
        if not self.should_run_stage(7, "intendedness"):
            return True

        self.logger.info("=" * 70)
        self.logger.info("Stage 7: Intendedness Classification")
        self.logger.info("=" * 70)

        self.logger.info("  7a: Build Input")
        log_file = self.get_stage_log(7, "build_input")

        metadata_dir = metadata_dir_for_type(BASE_DIR, self.pr_type)

        cmd = [
            "python3", str(BASE_DIR / "intendedness" / "build_intendedness_input.py"),
            "--repo", self.repo, "--pr-number", str(self.pr),
            "--base-dir", str(BASE_DIR),
            "--metadata-dir", str(metadata_dir),
            "--output-dir", str(self.output_base / "output"),
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        if not success:
            self.logger.error("      ✗ Failed")
            return False
        self.logger.info("      ✓ Completed")

        output_dir = self.output_base / "output" / self.repo
        self.logger.info("  7b: Run Classifier")
        log_file = self.get_stage_log(7, "classifier")
        cmd = [
            "python3", str(BASE_DIR / "intendedness" / "run_intendedness_classifier.py"),
            "--repo", self.repo, "--pr-number", str(self.pr),
            "--base-dir", str(BASE_DIR),
            "--input-dir", str(self.output_base / "output"),
            "--output-dir", str(output_dir),
        ]
        success, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        self.logger.info(f"      {'✓ Completed' if success else '✗ Failed'}")
        if not success:
            return False

        self.logger.info("  7c: Organize Results")
        log_file = self.get_stage_log(7, "organize")
        cmd = [
            "python3", str(BASE_DIR / "intendedness" / "organize_results.py"),
            "--repo", self.repo,
            "--pr-numbers", str(self.pr),
            "--base-dir", str(BASE_DIR),
            "--output-dir", str(output_dir),
        ]
        organized, _ = run_command_with_log(cmd, log_file)
        self.logger.info(f"      Log: {log_file}")
        self.logger.info(f"      {'✓ Completed' if organized else '✗ Failed'}")
        self.logger.info("")
        return success and organized

    def run(self) -> bool:
        """Run the full pipeline."""
        self.logger.info("=" * 70)
        self.logger.info(f"Starting {self.repo} PR #{self.pr}")
        self.logger.info(f"Start time: {datetime.now()}")
        self.logger.info("=" * 70)
        self.logger.info("")

        if not self.validate():
            return False

        stages = [
            (0, "Impact Region", self.run_stage_0),
            (1, "Intent Extraction", self.run_stage_1),
            (2, "Oracle Generation", self.run_stage_2),
            (3, "Target View", self.run_stage_3),
            (4, "Probe Generation", self.run_stage_4),
            (5, "Probe Runner", self.run_stage_5),
            (6, "Test Execution", self.run_stage_6),
            (7, "Intendedness", self.run_stage_7),
        ]

        for stage_num, stage_name, stage_func in stages:
            try:
                if not stage_func():
                    self.logger.error(f"✗ Stage {stage_num} ({stage_name}) failed")
                    return False
            except Exception as e:
                self.logger.error(f"✗ Stage {stage_num} ({stage_name}) raised an exception: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False

        self.logger.info("=" * 70)
        self.logger.info(f"✓ {self.repo} PR #{self.pr} completed")
        self.logger.info(f"End time: {datetime.now()}")
        self.logger.info(f"Main log: {self.main_log_file}")
        self.logger.info("=" * 70)
        return True



def get_all_prs(repo: str) -> List[int]:
    """Return all PR numbers found for a repository."""
    pr_dir = primary_repo_metadata_dir(BASE_DIR) / repo
    if not pr_dir.exists():
        return []

    prs = []
    for f in pr_dir.glob("pr_*.json"):
        try:
            prs.append(int(f.stem.split('_')[1]))
        except:
            continue
    return sorted(prs)


def get_all_repos() -> List[str]:
    """Return all repositories found under the default metadata root."""
    pr_dir = primary_repo_metadata_dir(BASE_DIR)
    if not pr_dir.exists():
        return []

    return sorted([d.name for d in pr_dir.iterdir() if d.is_dir()])


def main():
    parser = argparse.ArgumentParser(
        description="Full regression-analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py --repo marshmallow --pr 2699

  python3 pipeline.py --repo marshmallow

  python3 pipeline.py --all

  python3 pipeline.py --repo marshmallow --pr 2699 --force

Log locations: <ORION>/logs/
  - pipeline/<repo>/pr_<number>/run_<timestamp>.log  (main log)
  - stage<N>_<name>/<repo>/pr_<number>.log  (stage logs)
"""
    )

    parser.add_argument('--repo', type=str, help='Repository name')
    parser.add_argument('--pr', type=int, help='PR number')
    parser.add_argument('--all', action='store_true', help='Process every PR for every repository')
    parser.add_argument('--force', action='store_true', help='Force a full rerun and ignore resume state')

    args = parser.parse_args()

    if not args.all and not args.repo:
        parser.error("You must specify --repo or --all")

    if args.pr and not args.repo:
        parser.error("--pr requires --repo")

    tasks = []
    if args.all:
        for repo in get_all_repos():
            for pr in get_all_prs(repo):
                tasks.append((repo, pr))
    elif args.pr:
        tasks.append((args.repo, args.pr))
    else:
        for pr in get_all_prs(args.repo):
            tasks.append((args.repo, pr))

    print(f"=" * 70)
    print("Regression-analysis pipeline")
    print(f"=" * 70)
    print(f"Tasks: {len(tasks)}")
    print(f"Resume mode: {'disabled' if args.force else 'enabled'}")
    print(f"Log directory: {LOG_BASE_DIR}")
    print(f"=" * 70)
    print()

    success = 0
    failed = 0
    failed_list = []

    for repo, pr in tasks:
        pipeline = PRPipeline(repo, pr, args.force)
        try:
            if pipeline.run():
                success += 1
            else:
                failed += 1
                failed_list.append(f"{repo}#{pr}")
        except Exception as e:
            print(f"Exception: {e}")
            failed += 1
            failed_list.append(f"{repo}#{pr}")
        print()

    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Succeeded: {success}")
    print(f"Failed: {failed}")
    if failed_list:
        print("\nFailed items:")
        for item in failed_list:
            print(f"  - {item}")
    print(f"\nLogs: {LOG_BASE_DIR}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
