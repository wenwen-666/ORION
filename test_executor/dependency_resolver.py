#!/usr/bin/env python3
"""Dependency resolver for selecting and adjusting conda environments."""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BASE_ENV_TEMPLATES as CONFIG_BASE_ENV_TEMPLATES


class DependencyResolver:
    """Parse pyproject.toml and choose a suitable base environment."""

    BASE_ENV_TEMPLATES = CONFIG_BASE_ENV_TEMPLATES

    def __init__(self, base_dir: str = None, envs_root: Optional[str] = None):
        if base_dir is None:
            base_dir = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))
        self.base_dir = Path(base_dir)
        self.envs_root = Path(envs_root) if envs_root else None

    def parse_pyproject_toml(self, repo_path: Path) -> Dict[str, str]:
        pyproject_file = repo_path / 'pyproject.toml'
        if not pyproject_file.exists():
            return {}

        dependencies = {}
        try:
            content = pyproject_file.read_text()

            dep_pattern = r'"([a-zA-Z0-9_-]+)\s*([><=!]+[^"]+)"'
            matches = re.findall(dep_pattern, content)

            for package, constraint in matches:
                dependencies[package.lower()] = constraint.strip()

        except Exception as e:
            print(f"    [Warning] Failed to parse pyproject.toml: {e}")

        return dependencies

    def extract_numpy_version_range(self, constraint: str) -> Tuple[Optional[str], Optional[str]]:
        min_version = None
        max_version = None

        min_match = re.search(r'>=\s*([0-9.]+)', constraint)
        if min_match:
            min_version = min_match.group(1)

        max_match = re.search(r'<\s*([0-9.]+)', constraint)
        if max_match:
            max_version = max_match.group(1)

        return min_version, max_version

    def select_best_base_env(self, project: str, repo_path: Path,
                            commit_year: Optional[int] = None) -> Optional[Dict]:
        templates = self.BASE_ENV_TEMPLATES.get(project, [])
        if not templates:
            return None

        dependencies = self.parse_pyproject_toml(repo_path)
        numpy_constraint = dependencies.get('numpy', '')

        if numpy_constraint:
            min_ver, max_ver = self.extract_numpy_version_range(numpy_constraint)
            print(f"    [Dependency Analysis] NumPy constraint: {numpy_constraint}")
            print(f"    [Dependency Analysis] Parsed result: min={min_ver}, max={max_ver}")

            year_candidates = []
            if commit_year:
                for template in templates:
                    year_min, year_max = template['year_range']
                    if year_min <= commit_year <= year_max:
                        year_candidates.append(template)
            search_list = year_candidates if year_candidates else templates

            for template in search_list:
                template_numpy = template['numpy']

                if min_ver and self._compare_version(template_numpy, min_ver) < 0:
                    continue

                if max_ver and self._compare_version(template_numpy, max_ver) >= 0:
                    continue

                print(f"    [Environment Selection] Selected from dependency constraint: {template['name']}")
                return template

        if commit_year:
            for template in templates:
                year_min, year_max = template['year_range']
                if year_min <= commit_year <= year_max:
                    print(f"    [Environment Selection] Selected from commit year {commit_year}: {template['name']}")
                    return template

        if templates:
            print(f"    [Environment Selection] Using default environment: {templates[0]['name']}")
            return templates[0]

        return None

    def _compare_version(self, v1: str, v2: str) -> int:
        parts1 = [int(x) for x in v1.split('.')]
        parts2 = [int(x) for x in v2.split('.')]

        max_len = max(len(parts1), len(parts2))
        parts1.extend([0] * (max_len - len(parts1)))
        parts2.extend([0] * (max_len - len(parts2)))

        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1

        return 0

    def ensure_base_env_exists(self, template: Dict) -> bool:
        env_name = template['name']

        if self._env_exists(env_name):
            print(f"    [Base Env] {env_name} already exists")
            return True

        print(f"    [Base Env] Creating {env_name}...")
        try:
            create_cmd = ['conda', 'create']

            path_hint = self._get_env_prefix_path(env_name)
            if path_hint:
                path_hint.parent.mkdir(parents=True, exist_ok=True)
                create_cmd.extend(['-p', str(path_hint)])
            else:
                create_cmd.extend(['-n', env_name])

            create_cmd.append(f"python={template['python']}")

            if template.get('needs_compilers') or template.get('conda_packages'):
                conda_packages = template.get('conda_packages', [])
                create_cmd.extend(conda_packages)
                print(f"    [Base Env] Installing compilers: {', '.join(conda_packages)}")

            create_cmd.extend(['-y', '-q'])

            subprocess.run(
                create_cmd,
                capture_output=True,
                timeout=300,
                check=True
            )

            env_prefix = self._get_conda_prefix(env_name)
            pip_path = Path(env_prefix) / 'bin' / 'pip'

            for package in template['packages']:
                print(f"    [Base Env] Installing dependency: {package}")
                subprocess.run(
                    [str(pip_path), 'install', package, '-q'],
                    capture_output=True,
                    timeout=600,
                    check=True
                )

            build_tools = ['setuptools', 'wheel', 'pytest']
            if not template.get('needs_compilers'):
                build_tools.append('Cython')

            subprocess.run(
                [str(pip_path), 'install'] + build_tools + ['-q'],
                capture_output=True,
                timeout=300,
                check=True
            )

            print(f"    [Base Env] ✓ {env_name} created successfully")
            return True

        except Exception as e:
            print(f"    [Base Env] ✗ creation failed: {e}")
            return False

    def adjust_dependencies(self, env_name: str, repo_path: Path) -> bool:
        dependencies = self.parse_pyproject_toml(repo_path)
        if not dependencies:
            return True

        try:
            env_prefix = self._get_conda_prefix(env_name)
            pip_path = Path(env_prefix) / 'bin' / 'pip'
            python_path = Path(env_prefix) / 'bin' / 'python'

            for package, constraint in dependencies.items():
                if package not in ['numpy', 'cython']:
                    continue

                result = subprocess.run(
                    [str(python_path), '-c',
                     f'import {package}; print({package}.__version__)'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    continue

                current_version = result.stdout.strip()
                min_ver, max_ver = self.extract_numpy_version_range(constraint)

                needs_adjustment = False
                if min_ver and self._compare_version(current_version, min_ver) < 0:
                    needs_adjustment = True
                if max_ver and self._compare_version(current_version, max_ver) >= 0:
                    needs_adjustment = True

                if needs_adjustment:
                    print(f"    [Dependency Adjustment] {package} current version {current_version} does not satisfy constraint {constraint}")

                    subprocess.run(
                        [str(pip_path), 'uninstall', package, '-y', '-q'],
                        capture_output=True,
                        timeout=60
                    )

                    target_version = min_ver if min_ver else current_version
                    subprocess.run(
                        [str(pip_path), 'install', f'{package}=={target_version}', '-q'],
                        capture_output=True,
                        timeout=300,
                        check=True
                    )

                    print(f"    [Dependency Adjustment] ✓ {package} adjusted to {target_version}")

            return True

        except Exception as e:
            print(f"    [Dependency Adjustment] ✗ failed: {e}")
            return False

    def _get_env_prefix_path(self, env_name: str) -> Optional[Path]:
        """Return the configured prefix path for an environment."""
        if self.envs_root:
            return self.envs_root / env_name
        return None

    def _env_exists(self, env_name: str) -> bool:
        """Check whether a conda environment exists."""
        path_hint = self._get_env_prefix_path(env_name)
        if path_hint and (path_hint / 'conda-meta').exists():
            return True

        try:
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                env_data = json.loads(result.stdout)
                envs = env_data.get('envs', [])
                return any(Path(env_path).name == env_name for env_path in envs)
            return False
        except Exception:
            return False

    def _get_conda_prefix(self, env_name: str) -> str:
        """Return the filesystem path of a conda environment."""
        path_hint = self._get_env_prefix_path(env_name)
        if path_hint:
            python_path = path_hint / 'bin' / 'python'
            if python_path.exists():
                return str(path_hint)
            if path_hint.exists():
                return str(path_hint)

        result = subprocess.run(
            ['conda', 'env', 'list', '--json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            env_data = json.loads(result.stdout)
            envs = env_data.get('envs', [])
            for env_path in envs:
                if Path(env_path).name == env_name:
                    return env_path
        raise RuntimeError(f"Conda environment not found: {env_name}")

    def get_commit_year(self, repo_path: Path) -> Optional[int]:
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ci'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                date_str = result.stdout.strip()
                year = int(date_str.split('-')[0])
                return year
        except Exception as e:
            print(f"    [Warning] Failed to get commit year: {e}")
        return None


