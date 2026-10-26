#!/usr/bin/env python3
"""Create and manage isolated conda environments for pre- and post-PR commits."""

import subprocess
import json
import os
import tempfile
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_PYTHON_VERSION_BY_PROJECT,
    IMPORT_NAME_MAPPING,
    PROJECT_EXTRA_DEPENDENCIES,
    PROJECT_PATH_MAPPING,
    PYTHON_PACKAGE_SUBDIR,
    SKIP_IMPORT_VALIDATION_PROJECTS,
    VERIFY_IMPORT_NAME_MAPPING,
    resolve_project_base,
    resolve_project_checkout,
)


class CondaEnvManager:
    """Manage per-PR conda environments for test execution."""

    MAX_FETCH_RETRIES = 3

    def __init__(self, base_dir: str = None):
        """
        Initialize the conda environment manager.

        Args:
            base_dir: Workspace root containing the project checkouts.
        """
        if base_dir is None:
            base_dir = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))
        self.base_dir = Path(base_dir)
        self.envs_cache = {}
        self.project_base = resolve_project_base(self.base_dir)

        self.envs_root = self.base_dir / "conda_envs"
        self.envs_root.mkdir(parents=True, exist_ok=True)

        self.work_dir = self.base_dir / "conda_env_workspace"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.project_path_mapping = PROJECT_PATH_MAPPING
        self.python_package_subdir = PYTHON_PACKAGE_SUBDIR
        self.import_name_mapping = IMPORT_NAME_MAPPING

        from dependency_resolver import DependencyResolver
        self.resolver = DependencyResolver(base_dir, envs_root=str(self.envs_root))
        self.project_extra_dependencies = PROJECT_EXTRA_DEPENDENCIES


    def _check_disk_space(self, required_gb: float = 5.0) -> bool:
        import shutil
        try:
            stat = shutil.disk_usage(self.envs_root)
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_gb = stat.used / (1024**3)

            if free_gb < required_gb:
                raise RuntimeError(
                    f"Insufficient disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total "
                    f"(used {used_gb:.1f}GB); at least {required_gb}GB is required"
                )

            if free_gb < 10:
                print(f"    [Warning] Low disk space: {free_gb:.1f}GB remaining")

            return True
        except Exception as e:
            print(f"    [Warning] Unable to check disk space: {e}")
            return True

    def setup_pr_environments(self, project: str, pr_number: int,
                             pre_commit: str, post_commit: str,
                             progress_callback=None) -> Tuple[str, str]:
        if progress_callback:
            progress_callback(f"Starting setup for {project} PR#{pr_number}")

        print(f"\n[Environment Setup] PR #{pr_number}")
        print("-" * 70)

        pre_env_name = f"{project}_pr{pr_number}_pre"
        post_env_name = f"{project}_pr{pr_number}_post"

        pre_env_path = self._create_or_get_env(project, pre_commit, pre_env_name)
        if progress_callback:
            progress_callback(f"Pre environment ready: {pre_env_name}")

        post_env_path = self._create_or_get_env(project, post_commit, post_env_name)
        if progress_callback:
            progress_callback(f"Post environment ready: {post_env_name}")

        print(f"  ✓ Pre-commit environment: {pre_env_name}")
        print(f"  ✓ Post-commit environment: {post_env_name}")

        if progress_callback:
            progress_callback(f"Finished setting up {project} PR#{pr_number}")

        return pre_env_name, post_env_name

    def _get_env_repo_dir(self, env_name: str) -> Path:
        """Return the source checkout directory for an environment."""
        return self.work_dir / env_name / 'repo'

    def _prepare_temp_repo(self, project_path: Path, commit: str) -> Path:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix='conda_env_analysis_'))
        temp_repo = temp_dir / 'repo'

        try:
            subprocess.run(
                ['git', 'clone', str(project_path), str(temp_repo)],
                capture_output=True,
                timeout=60,
                check=True
            )

            subprocess.run(
                ['git', 'checkout', commit],
                cwd=temp_repo,
                capture_output=True,
                timeout=30,
                check=True
            )

            return temp_repo

        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to prepare temporary repository: {e}")

    def _create_or_get_env(self, project: str, commit: str, env_name: str) -> str:

        try:
            self._check_disk_space(required_gb=5.0)
        except RuntimeError as e:
            print(f"    [Error] {e}")
            raise

        if self._env_exists(env_name):
            expected_python = self._get_python_version_for_project(project)
            env_python = self._get_env_prefix_path(env_name) / 'bin' / f'python{expected_python}'
            python_ok = env_python.exists()
            if self._verify_env_commit(env_name, commit) and python_ok:
                print(f"  ⚡ Reusing environment: {env_name} (commit: {commit[:8]})")
                self._ensure_test_tooling(env_name)
                self._install_project_extras(project, env_name)
                return env_name
            else:
                reason = "commit mismatch" if not self._verify_env_commit(env_name, commit) else f"Python version mismatch (expected {expected_python})"
                print(f"  🔄 {reason}; rebuilding: {env_name}")
                self.cleanup_env(env_name)
                env_work_dir = self.work_dir / env_name
                if env_work_dir.exists():
                    shutil.rmtree(env_work_dir)

        print(f"  🔨 Creating environment: {env_name} (commit: {commit[:8]})")

        project_path = resolve_project_checkout(self.project_base, project)
        if not project_path.exists():
            project_relative_path = self.project_path_mapping.get(project, project)
            fallback_path = self.base_dir / project_relative_path
            if fallback_path.exists():
                project_path = fallback_path

        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")

        commit_year = None

        try:
            self._ensure_commit_available(project_path, commit)

            temp_repo = self._prepare_temp_repo(project_path, commit)
            commit_year = self.resolver.get_commit_year(temp_repo)
            if commit_year:
                print(f"    [Analysis] Commit year: {commit_year}")
            shutil.rmtree(temp_repo.parent)

            self._create_base_env(env_name, project, commit_year)
            self._install_project_at_commit(env_name, project_path, commit)

        except Exception as e:
            print(f"    [Error] Environment creation failed: {e}")
            self.cleanup_env(env_name)
            raise RuntimeError(f"Environment creation failed: {e}")

        self._save_env_commit(env_name, commit)

        self._ensure_test_tooling(env_name)
        self._install_project_extras(project, env_name)

        print(f"    [Validation] Checking environment integrity...")
        python_bin = self._get_env_prefix_path(env_name) / "bin" / "python"
        if not python_bin.exists():
            print(f"    [Error] Python executable not found")
            self.cleanup_env(env_name)
            raise RuntimeError(f"Environment creation failed: Python executable not found")

        print(f"    ✓ Environment validation passed")

        self.envs_cache[env_name] = {
            'project': project,
            'commit': commit,
            'path': project_path
        }

        return env_name

    def _get_env_prefix_path(self, env_name: str) -> Path:
        """Return the target prefix path for a conda environment."""
        return self.envs_root / env_name

    def _env_exists(self, env_name: str) -> bool:
        """Check whether a conda environment exists."""
        prefix_path = self._get_env_prefix_path(env_name)
        if (prefix_path / 'conda-meta').exists():
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
        except Exception as e:
            print(f"    [Warning] Environment check failed: {e}")
            return False

    def _create_base_env(self, env_name: str, project: str, commit_year: Optional[int] = None):
        """Create a base conda environment, including compilers and BLAS when needed."""

        python_version = self._get_python_version_for_project(project, commit_year)

        env_prefix = self._get_env_prefix_path(env_name)
        env_prefix.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'conda', 'create', '-p', str(env_prefix),
            f'python={python_version}',
            'gcc_linux-64',
            'gxx_linux-64',
            'gfortran_linux-64',
            'openblas',
            'libgfortran',
            'cmake',
            'make',
            '-y', '-q'
        ]

        print(f"    [Environment Setup] Creating environment (Python {python_version}) and installing compilers and math libraries...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create conda environment: {result.stderr}")

    def _get_python_version_for_project(self, project: str, commit_year: Optional[int] = None) -> str:

        return DEFAULT_PYTHON_VERSION_BY_PROJECT.get(project, '3.10')

    def _install_build_dependencies(self, env_name: str, commit_year: Optional[int] = None):
        print(f"    [Dependency Injection] Pre-installing core build dependencies...")

        env_prefix = self._get_conda_prefix(env_name)
        pip_path = Path(env_prefix) / 'bin' / 'pip'

        build_deps = [
            'setuptools',
            'wheel',
            'cython<3',
            'numpy<2',
        ]

        if commit_year:
            if commit_year <= 2023:
                build_deps = [
                    'setuptools<70',
                    'wheel<0.42',
                    'cython<0.30',
                    'numpy<1.25',
                ]
            elif commit_year == 2024:
                build_deps = [
                    'setuptools<71',
                    'wheel<0.43',
                    'cython<3',
                    'numpy<1.27',
                ]

        print(f"    [Dependency Injection] Installing: {', '.join(build_deps)}")

        max_retries = 3
        timeout = 900

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"    [Retry] Attempt {attempt + 1}...")

                result = subprocess.run(
                    [str(pip_path), 'install'] + build_deps + ['-q'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0:
                    print(f"    ✓ Build dependencies installed")
                    return
                else:
                    print(f"    [Warning] Failed to install build dependencies (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(5)
                    else:
                        print(f"    [Error] Failed to install build dependencies: {result.stderr[-500:]}")

            except subprocess.TimeoutExpired:
                print(f"    [Warning] Build dependency installation timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
                else:
                    print(f"    [Error] Build dependency installation timed out after {max_retries} retries")
            except Exception as e:
                print(f"    [Warning] Build dependency installation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"    [Error] Failed to install build dependencies after {max_retries} retries")

        print(f"    [Warning] Failed to install build dependencies; continuing with project installation...")

    def _adjust_numpy_for_compilation(self, env_name: str, project: str, commit_year: int):
        print(f"    [ABI Fix] Adjusting NumPy version to match {project} in {commit_year}...")

        if commit_year <= 2019:
            target_numpy = 'numpy>=1.15,<1.18'
        elif commit_year <= 2020:
            target_numpy = 'numpy>=1.17,<1.20'
        elif commit_year <= 2021:
            target_numpy = 'numpy>=1.19,<1.22'
        elif commit_year <= 2022:
            target_numpy = 'numpy>=1.21,<1.24'
        elif commit_year <= 2023:
            target_numpy = 'numpy>=1.23,<1.26'
        else:
            target_numpy = 'numpy>=1.24,<2.0'

        print(f"    [ABI Fix] Target NumPy version: {target_numpy}")

        env_prefix = self._get_conda_prefix(env_name)
        pip_path = Path(env_prefix) / 'bin' / 'pip'

        try:
            result = subprocess.run(
                [str(pip_path), 'install', target_numpy, '--force-reinstall', '-q'],
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode != 0:
                print(f"    [Warning] Failed to adjust NumPy version: {result.stderr}")
            else:
                verify_result = subprocess.run(
                    [str(pip_path), 'show', 'numpy'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if verify_result.returncode == 0:
                    for line in verify_result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            numpy_version = line.split(':')[1].strip()
                            print(f"    ✓ NumPy version adjusted to: {numpy_version}")
                            break

        except Exception as e:
            print(f"    [Warning] NumPy adjustment error: {e}")

    def _ensure_commit_available(self, repo_path: Path, commit: str) -> None:
        """Ensure the requested commit exists locally, fetching it if needed."""
        if not commit:
            return
        already_present = self._has_git_object(repo_path, commit)
        if not already_present:
            short = commit[:8]
            print(f"    [Repository Update] Local repository is missing commit {short}; attempting to fetch...")

            if not self._fetch_commit(repo_path, commit, depth=1):
                print(f"    [Repository Update] Shallow fetch failed; retrying with a full fetch for {short}...")
                if not self._fetch_commit(repo_path, commit, depth=None):
                    raise RuntimeError(f"Unable to fetch commit {commit}; please check the remote repository or network connection")

        self._ensure_local_ref(repo_path, commit)

    def _has_git_object(self, repo_path: Path, commit: str) -> bool:
        """Check whether the repository already contains the requested object."""
        result = subprocess.run(
            ['git', 'cat-file', '-e', commit],
            cwd=repo_path,
            capture_output=True
        )
        return result.returncode == 0

    def _fetch_commit(self, repo_path: Path, commit: str, depth: Optional[int]) -> bool:
        """Fetch a commit with optional depth and retry logic."""
        depth_flag = [f'--depth={depth}'] if depth else []
        for attempt in range(1, self.MAX_FETCH_RETRIES + 1):
            cmd = ['git', 'fetch', *depth_flag, 'origin', commit]
            print(
                f"    [Repository Update] fetch {commit[:8]} "
                f"(depth={'full' if depth is None else depth}, attempt {attempt})..."
            )
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return True
            print(f"      ↩ fetch failed: {result.stderr.strip()}")
            time.sleep(1)
        return False

    def _ensure_local_ref(self, repo_path: Path, commit: str) -> None:
        """Create a stable tag so later clones can resolve the commit."""
        ref_name = f"refs/tags/pr_cache_{commit}"
        show_ref = subprocess.run(
            ['git', 'show-ref', '--verify', '--quiet', ref_name],
            cwd=repo_path,
            capture_output=True
        )
        if show_ref.returncode == 0:
            return
        subprocess.run(
            ['git', 'update-ref', ref_name, commit],
            cwd=repo_path,
            check=False,
            capture_output=True
        )

    def _install_project_at_commit(self, env_name: str, project_path: Path, commit: str):

        env_work_dir = self.work_dir / env_name
        env_work_dir.mkdir(parents=True, exist_ok=True)

        repo_path = env_work_dir / 'repo'

        if repo_path.exists():
            shutil.rmtree(repo_path)

        try:
            self._ensure_commit_available(project_path, commit)

            clone_result = subprocess.run(
                ['git', 'clone', str(project_path), str(repo_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if clone_result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {clone_result.stderr}")

            checkout_result = subprocess.run(
                ['git', 'checkout', commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if checkout_result.returncode != 0:
                raise RuntimeError(f"Failed to checkout commit: {checkout_result.stderr}")

            print(f"    [Cleanup] Removing all pr_cache tags...")
            list_tags_result = subprocess.run(
                ['git', 'tag', '-l', 'pr_cache_*'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if list_tags_result.returncode == 0 and list_tags_result.stdout.strip():
                pr_cache_tags = list_tags_result.stdout.strip().split('\n')
                for tag in pr_cache_tags:
                    subprocess.run(
                        ['git', 'tag', '-d', tag],
                        cwd=repo_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                print(f"    ✓ Removed {len(pr_cache_tags)} pr_cache tags")

            gitmodules_file = repo_path / '.gitmodules'
            if gitmodules_file.exists():
                project_name_lower = project_path.name.lower()
                print(f"    [Submodule] Detected .gitmodules; attempting to initialize submodules...")
                submodule_copied = False
                if project_name_lower == 'scipy':
                    scipy_submodules = (
                        'subprojects/xsf', 'scipy/_lib/array_api_compat',
                        'scipy/_lib/pocketfft', 'scipy/_lib/array_api_extra',
                        'scipy/_lib/cobyqa', 'scipy/_lib/unuran',
                        'subprojects/boost_math/math', 'subprojects/highs',
                    )
                    subprocess.run(['git', 'submodule', 'init'],
                                   cwd=repo_path, capture_output=True, timeout=10)
                    for sub_rel in scipy_submodules:
                        src = project_path / sub_rel
                        if src.exists() and any(src.iterdir()):
                            subprocess.run(
                                ['git', 'config', f'submodule.{sub_rel}.url', str(src)],
                                cwd=repo_path, capture_output=True, timeout=5
                            )
                    subprocess.run(
                        ['git', 'submodule', 'update', '--init', '--recursive'],
                        cwd=repo_path, capture_output=True, timeout=60
                    )
                    submodule_copied = True
                    print(f"    ✓ Submodules initialized from the main repository")
                if not submodule_copied:
                    try:
                        submodule_result = subprocess.run(
                            ['git', 'submodule', 'update', '--init', '--recursive'],
                            cwd=repo_path, capture_output=True, text=True, timeout=60
                        )
                        if submodule_result.returncode == 0:
                            print(f"    ✓ Submodule initialization completed")
                        else:
                            print(f"    [Warning] Submodule initialization failed (skipped): {submodule_result.stderr[:200]}")
                    except Exception as e:
                        print(f"    [Warning] Submodule initialization timed out/failed (skipped): {e}")

            commit_year = None
            try:
                year_result = subprocess.run(
                    ['git', 'show', '-s', '--format=%ci', commit],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if year_result.returncode == 0:
                    commit_year = int(year_result.stdout.strip()[:4])
                    print(f"    [Analysis] Commit year: {commit_year}")
            except Exception:
                pass

            conda_prefix = self._get_conda_prefix(env_name)
            pip_path = Path(conda_prefix) / 'bin' / 'pip'
            python_path = Path(conda_prefix) / 'bin' / 'python'

            self._install_build_dependencies(env_name, commit_year)

            project_name = project_path.name

            if project_name.lower() in ('scipy', 'numpy'):
                conda_prefix = self._get_conda_prefix(env_name)
                pip_path = Path(conda_prefix) / 'bin' / 'pip'
                print(f"    [Dependency Injection] Installing the meson build backend (meson-python, meson, ninja, pybind11, cython>=3, pythran)...")
                subprocess.run(
                    [str(pip_path), 'install', 'meson-python', 'meson', 'ninja', 'pybind11', 'cython>=3.0.8', 'pythran', '-q'],
                    capture_output=True, timeout=300
                )
            self._run_source_generation_scripts(repo_path, project_name, python_path)

            has_cython = bool(list(repo_path.glob('**/*.pyx')))
            compile_projects = {'pandas', 'scipy', 'numpy', 'aiohttp', 'polars'}
            needs_compilation = has_cython or project_name.lower() in compile_projects

            if project_name.lower() == 'polars':
                install_timeout = 1800
            elif needs_compilation:
                install_timeout = 900
            else:
                install_timeout = 600

            print(f"    [Install] pytest...")
            subprocess.run(
                [str(pip_path), 'install', 'pytest', '-q'],
                timeout=120,
                capture_output=True
            )

            req_file = repo_path / 'requirements.txt'
            if req_file.exists():
                print(f"    [Install] requirements.txt (with version limits)...")
                self._install_requirements_with_constraints(pip_path, req_file, commit_year, repo_path)

            dev_req_candidates = [
                "requirements-dev.txt",
                "requirements-test.txt",
                "dev-requirements.txt",
                "test-requirements.txt",
                "requirements/dev.txt",
                "requirements/test.txt",
            ]
            print(f"    [Dependency Discovery] Looking for development dependency files...")
            for req_name in dev_req_candidates:
                req_path = repo_path / req_name
                if not req_path.exists():
                    continue
                print(f"    [Install] Found development dependency: {req_name}, installing...")
                try:
                    subprocess.run(
                        [str(pip_path), 'install', '-r', str(req_path), '-q'],
                        cwd=repo_path,
                        timeout=300,
                        capture_output=True
                    )
                except Exception as exc:
                    print(f"    [Warning] Failed to install {req_name}: {exc}")

            install_dir = repo_path

            if project_name == 'langchain':
                install_env = os.environ.copy()
                install_env['SETUPTOOLS_SCM_PRETEND_VERSION'] = '0.0.0.dev0'
                langchain_subdirs = ['libs/core', 'libs/text-splitters', 'libs/langchain']
                print(f"    [langchain] monorepo multi-package installation...")
                for sub in langchain_subdirs:
                    sub_path = repo_path / sub
                    if not sub_path.exists():
                        print(f"    [langchain] Skipping missing subdirectory: {sub}")
                        continue
                    print(f"    [langchain] Installing subpackage: {sub}")
                    r = subprocess.run(
                        [str(pip_path), 'install', '-e', '.', '-q'],
                        cwd=sub_path,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        env=install_env
                    )
                    if r.returncode != 0:
                        print(f"    [Warning] {sub} installation failed: {r.stderr[:200]}")
                    else:
                        print(f"    ✓ {sub} installation completed")
                print(f"    ✓ langchain monorepo installation completed")
            elif project_name in self.python_package_subdir:
                subdir = self.python_package_subdir[project_name]
                install_dir = repo_path / subdir
                print(f"    [Note] Python package lives in subdirectory: {subdir}")
            if project_name == 'langchain':
                install_cmd = None
            elif needs_compilation:
                print(f"    [Install] Installing project package (compiled mode, preserving historical dependencies)...")
                install_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--no-build-isolation',
                    '-q'
                ]
                fallback_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '-q'
                ]
            else:
                print(f"    [Install] Installing project package (pure-Python standard mode)...")
                install_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '-q'
                ]
                fallback_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--no-deps',
                    '-q'
                ]

            install_env = os.environ.copy()
            install_env['SETUPTOOLS_SCM_PRETEND_VERSION'] = '0.0.0.dev0'

            conda_prefix = self._get_conda_prefix(env_name)
            activate_d = Path(conda_prefix) / 'etc' / 'conda' / 'activate.d'
            if activate_d.exists():
                activate_env_str = f'export CONDA_PREFIX={conda_prefix}\n'
                for sh_file in sorted(activate_d.glob('*.sh')):
                    activate_env_str += f'source {sh_file}\n'
                activate_env_str += 'env'
                try:
                    result = subprocess.run(
                        ['bash', '-c', activate_env_str],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if '=' in line:
                                k, _, v = line.partition('=')
                                if k in ('CC', 'CXX', 'FC', 'F77', 'F90', 'LD', 'AR', 'NM',
                                         'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'FFLAGS',
                                         'GCC', 'GXX', 'GFORTRAN',
                                         'HOST', 'BUILD', 'PREFIX',
                                         'CONDA_PREFIX', 'CONDA_BUILD_SYSROOT'):
                                    install_env[k] = v
                except Exception:
                    pass
            conda_bin = str(Path(conda_prefix) / 'bin')
            install_env['PATH'] = conda_bin + ':' + install_env.get('PATH', '')

            if install_cmd is not None:
                install_result = subprocess.run(
                    install_cmd,
                    cwd=install_dir,
                    capture_output=True,
                    text=True,
                    timeout=install_timeout,
                    env=install_env
                )

                if install_result.returncode != 0:
                    print(f"    [Warning] No-isolation install failed; retrying with a standard install...")
                    fallback_result = subprocess.run(
                        fallback_cmd,
                        cwd=install_dir,
                        capture_output=True,
                        text=True,
                        timeout=install_timeout,
                        env=install_env
                    )

                    if fallback_result.returncode != 0:
                        print(f"    [Error] Project installation failed: {install_result.stderr}")
                        print(f"    [Error] Fallback installation also failed: {fallback_result.stderr}")
                        raise RuntimeError(f"Project installation failed")
                else:
                    print(f"    ✓ Project installation completed")

            if project_name.lower() == 'keras':
                self._install_keras_backend(env_name, repo_path, commit_year)

            self._verify_installation(python_path, project_name)

            compile_projects = {'pandas', 'scipy', 'numpy', 'aiohttp'}
            if project_name.lower() in compile_projects:
                self._verify_compilation(python_path, project_name)

        except Exception as e:
            print(f"    [Error] Installation process error: {e}")
            if env_work_dir.exists():
                shutil.rmtree(env_work_dir)
            raise

    def _clone_env(self, source_env: str, target_env: str):
        print(f"    Cloning environment: {source_env} -> {target_env}")

        source_prefix = self._get_conda_prefix(source_env)
        target_prefix = self._get_env_prefix_path(target_env)
        target_prefix.parent.mkdir(parents=True, exist_ok=True)
        if target_prefix.exists():
            shutil.rmtree(target_prefix)

        cmd = [
            'conda', 'create', '-p', str(target_prefix),
            '--clone', source_prefix,
            '-y', '-q'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone environment: {result.stderr}")

        print(f"    ✓ Clone completed (all dependencies were copied)")

        try:
            env_prefix = self._get_conda_prefix(target_env)
            python_path = Path(env_prefix) / 'bin' / 'python'

            print(f"    [Optimization] Upgrading pip...")

            result = subprocess.run(
                [str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip', '-q'],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"    [Info] Standard upgrade failed; trying get-pip.py...")

                version_result = subprocess.run(
                    [str(python_path), '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                python_version = version_result.stdout.strip() if version_result.returncode == 0 else '3.10'

                if python_version.startswith('3.6'):
                    get_pip_url = 'https://bootstrap.pypa.io/pip/3.6/get-pip.py'
                else:
                    get_pip_url = 'https://bootstrap.pypa.io/get-pip.py'

                import urllib.request
                get_pip_path = '/tmp/get-pip.py'
                urllib.request.urlretrieve(get_pip_url, get_pip_path)

                subprocess.run(
                    [str(python_path), get_pip_path, '-q'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                Path(get_pip_path).unlink(missing_ok=True)

            print(f"    ✓ pip upgraded")
        except Exception as e:
            print(f"    [Warning] pip upgrade failed: {e}")

    def _run_source_generation_scripts(self, repo_path: Path, project_name: str, python_path: Path):
        pyx_files = list(repo_path.glob('**/*.pyx'))
        if pyx_files:
            print(f"    [Cython Precompile] Detected {len(pyx_files)} .pyx files; starting conversion...")
            self._compile_cython_files(repo_path, python_path, pyx_files)

        if project_name == 'aiohttp':
            gen_script = repo_path / 'tools' / 'gen.py'
            if gen_script.exists():
                print(f"    [Source Preprocessing] Running tools/gen.py...")
                try:
                    result = subprocess.run(
                        [str(python_path), str(gen_script)],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        print(f"    ✓ Code generation completed")
                    else:
                        print(f"    [Warning] Code generation failed (continuing): {result.stderr[:200]}")
                except Exception as e:
                    print(f"    [Warning] Code generation error (continuing): {e}")

        common_gen_scripts = [
            'generate_code.py',
            'codegen.py',
            'build_ext.py',
        ]

        for script_name in common_gen_scripts:
            script_path = repo_path / script_name
            if script_path.exists():
                print(f"    [Source Preprocessing] Running {script_name}...")
                try:
                    subprocess.run(
                        [str(python_path), str(script_path)],
                        cwd=repo_path,
                        capture_output=True,
                        timeout=60
                    )
                except Exception as e:
                    print(f"    [Warning] {script_name} execution failed (continuing): {e}")

    def _compile_cython_files(self, repo_path: Path, python_path: Path, pyx_files: list):
        try:
            success_count = 0
            fail_count = 0

            for pyx_file in pyx_files:
                try:
                    result = subprocess.run(
                        [str(python_path), '-m', 'cython', str(pyx_file)],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode == 0:
                        success_count += 1
                        c_file = pyx_file.with_suffix('.c')
                        if c_file.exists():
                            print(f"      ✓ {pyx_file.relative_to(repo_path)} -> {c_file.name}")
                        else:
                            print(f"      [Warning] {pyx_file.relative_to(repo_path)} compiled successfully but did not generate a .c file")
                    else:
                        fail_count += 1
                        print(f"      [Warning] {pyx_file.relative_to(repo_path)} compilation failed: {result.stderr[:100]}")

                except subprocess.TimeoutExpired:
                    fail_count += 1
                    print(f"      [Warning] {pyx_file.relative_to(repo_path)} compilation timed out")
                except Exception as e:
                    fail_count += 1
                    print(f"      [Warning] {pyx_file.relative_to(repo_path)} compilation error: {e}")

            if success_count > 0:
                print(f"    ✓ Cython precompilation completed: {success_count} succeeded, {fail_count} failed")
            else:
                print(f"    [Warning] All Cython precompilation attempts failed; letting pip handle it")

        except Exception as e:
            print(f"    [Warning] Cython precompilation error: {e}")

    def _install_requirements_with_constraints(self, pip_path: Path, req_file: Path,
                                               commit_year: Optional[int], repo_path: Path):
        try:
            with open(req_file, 'r') as f:
                requirements = f.readlines()

            if commit_year:
                constrained_req = repo_path / 'requirements_constrained.txt'

                with open(constrained_req, 'w') as f:
                    for line in requirements:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            f.write(line + '\n')
                            continue

                        if '<' not in line and '==' not in line and commit_year <= 2024:
                            pkg_name = line.split('>')[0].split('[')[0].strip()

                            if pkg_name.lower() in ['numpy', 'pandas', 'scipy']:
                                if commit_year <= 2023:
                                    line = f"{pkg_name}<2.0"
                                else:
                                    line = f"{pkg_name}<2.2"

                        f.write(line + '\n')

                subprocess.run(
                    [str(pip_path), 'install', '-r', str(constrained_req), '-q'],
                    cwd=repo_path,
                    timeout=300,
                    capture_output=True
                )

                constrained_req.unlink()
            else:
                subprocess.run(
                    [str(pip_path), 'install', '-r', str(req_file), '-q'],
                    cwd=repo_path,
                    timeout=300,
                    capture_output=True
                )

        except Exception as e:
            print(f"    [Warning] requirements installation failed: {e}")

    def _verify_installation(self, python_path: Path, project_name: str):
        print(f"    [Strict Validation] Verifying whether {project_name} can be imported...")

        import_name = self.import_name_mapping.get(project_name, project_name)

        skip_import_projects = SKIP_IMPORT_VALIDATION_PROJECTS

        if project_name in skip_import_projects or import_name in skip_import_projects or import_name is None:
            print(f"    ⊘ {project_name} skipping import validation (special-case project)")
            return

        try:
            result = subprocess.run(
                [str(python_path), '-c', f'import {import_name}; print("{import_name} imported successfully")'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print(f"    ✓ Strict validation passed: {import_name} imports successfully")
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                print(f"    ✗ Strict validation failed: {error_msg}")
                print(f"    [Warning] {project_name} import failed, but continuing (this project may need runtime configuration)")

        except subprocess.TimeoutExpired:
            print(f"    ✗ Strict validation timed out")
            print(f"    [Warning] {project_name} import timed out, but continuing")
        except Exception as e:
            print(f"    ✗ Strict validation error: {e}")
            print(f"    [Warning] {project_name} validation raised an exception, but continuing")

    def _verify_compilation(self, python_path: Path, project_name: str):
        print(f"    [Build Validation] Checking C extensions for {project_name}...")

        verification_code = {
            'pandas': """
import pandas as pd
import pandas._libs.lib as lib
import pandas._libs.algos as algos
print('Pandas C extensions loaded successfully')
print(f'pandas version: {pd.__version__}')
""",
            'scipy': """
import scipy
import scipy.linalg
import scipy.sparse
print('Scipy C extensions loaded successfully')
print(f'scipy version: {scipy.__version__}')
""",
            'numpy': """
import numpy as np
import numpy.core._multiarray_umath
print('Numpy C extensions loaded successfully')
print(f'numpy version: {np.__version__}')
""",
            'aiohttp': """
import aiohttp
from aiohttp import _http_parser
print('aiohttp C extensions loaded successfully')
print(f'aiohttp version: {aiohttp.__version__}')
"""
        }

        if project_name not in verification_code:
            print(f"    ⊘ {project_name} does not require C-extension validation")
            return

        test_code = verification_code[project_name]

        try:
            result = subprocess.run(
                [str(python_path), '-c', test_code],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                print(f"    ✓ C-extension validation passed")
                for line in result.stdout.strip().split('\n'):
                    if 'version' in line.lower():
                        print(f"      {line}")
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                print(f"    ✗ C-extension validation failed: {error_msg}")
                raise RuntimeError(f"Failed to load C extension: {error_msg}")

        except subprocess.TimeoutExpired:
            print(f"    ✗ C-extension validation timed out")
            raise RuntimeError(f"C-extension validation timed out")
        except Exception as e:
            print(f"    ✗ C-extension validation error: {e}")
            raise RuntimeError(f"C-extension validation error: {e}")

    def health_check(self, env_name: str, project_name: str = None) -> Dict[str, Any]:
        print(f"    [Health Check] Checking environment {env_name}...")

        health_status = {
            'env_name': env_name,
            'python_version': None,
            'numpy_version': None,
            'compiler_available': False,
            'blas_library': None,
            'project_installed': None,
        }

        try:
            python_path = self._get_python_path(env_name)

            try:
                result = subprocess.run(
                    [str(python_path), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version_str = result.stdout.strip() or result.stderr.strip()
                    health_status['python_version'] = version_str.replace('Python ', '')
                    print(f"      ✓ Python version: {health_status['python_version']}")
            except Exception as e:
                print(f"      ✗ Python version check failed: {e}")

            try:
                result = subprocess.run(
                    [str(python_path), '-c', 'import numpy; print(numpy.__version__)'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    health_status['numpy_version'] = result.stdout.strip()
                    print(f"      ✓ NumPy version: {health_status['numpy_version']}")
                else:
                    print(f"      - NumPy not installed")
            except Exception as e:
                print(f"      ✗ NumPy version check failed: {e}")

            env_prefix = self._get_conda_prefix(env_name)
            compilers = {
                'gcc': Path(env_prefix) / 'bin' / 'x86_64-conda-linux-gnu-gcc',
                'g++': Path(env_prefix) / 'bin' / 'x86_64-conda-linux-gnu-g++',
                'gfortran': Path(env_prefix) / 'bin' / 'x86_64-conda-linux-gnu-gfortran',
            }

            compiler_status = {}
            for name, path in compilers.items():
                compiler_status[name] = path.exists()

            health_status['compiler_available'] = compiler_status
            all_available = all(compiler_status.values())
            if all_available:
                print(f"      ✓ Compilers: gcc, g++, and gfortran are all available")
            else:
                missing = [k for k, v in compiler_status.items() if not v]
                print(f"      ⚠ Missing compilers: {', '.join(missing)}")

            try:
                result = subprocess.run(
                    [str(python_path), '-c',
                     'import numpy; config = numpy.__config__.show(); print(config)'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'openblas' in output:
                        health_status['blas_library'] = 'openblas'
                        print(f"      ✓ BLAS library: OpenBLAS")
                    elif 'mkl' in output:
                        health_status['blas_library'] = 'mkl'
                        print(f"      ✓ BLAS library: MKL")
                    elif 'blas' in output:
                        health_status['blas_library'] = 'blas'
                        print(f"      ✓ BLAS library: system BLAS")
                    else:
                        print(f"      - BLAS library: not detected")
            except Exception as e:
                print(f"      ✗ BLAS library check failed: {e}")

            if project_name:
                try:
                    result = subprocess.run(
                        [str(python_path), '-c', f'import {project_name}; print({project_name}.__version__)'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        health_status['project_installed'] = version
                        print(f"      ✓ {project_name} installed: {version}")
                    else:
                        health_status['project_installed'] = False
                        print(f"      - {project_name} not installed")
                except Exception as e:
                    health_status['project_installed'] = False
                    print(f"      ✗ {project_name} check failed: {e}")

            print(f"    [Health Check] Completed")
            return health_status

        except Exception as e:
            print(f"    [Health Check] Failed: {e}")
            health_status['error'] = str(e)
            return health_status

    def _install_keras_backend(self, env_name: str, repo_path: Path, commit_year: Optional[int] = None):
        print(f"    [Keras Backend] Detecting and installing TensorFlow...")

        env_prefix = self._get_conda_prefix(env_name)
        pip_path = Path(env_prefix) / 'bin' / 'pip'
        python_path = Path(env_prefix) / 'bin' / 'python'

        tf_version = self._parse_tensorflow_version_from_setup(repo_path)

        if not tf_version:
            tf_version = self._infer_tensorflow_version(repo_path, commit_year)

        if tf_version:
            print(f"    [Keras Backend] Installing TensorFlow {tf_version}...")
            try:
                result = subprocess.run(
                    [str(pip_path), 'install', tf_version, '-q'],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode == 0:
                    print(f"    ✓ TensorFlow installation completed")

                    verify_result = subprocess.run(
                        [str(python_path), '-c', 'import tensorflow as tf; print(f"TensorFlow {tf.__version__}")'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if verify_result.returncode == 0:
                        print(f"    ✓ TensorFlow validation succeeded: {verify_result.stdout.strip()}")
                    else:
                        print(f"    ⚠ TensorFlow validation failed: {verify_result.stderr}")
                else:
                    print(f"    ⚠ TensorFlow installation failed: {result.stderr}")
                    print(f"    [Note] Keras may not function correctly")

            except Exception as e:
                print(f"    ⚠ TensorFlow installation error: {e}")
        else:
            print(f"    ⚠ Could not determine the TensorFlow version; skipping backend installation")

        self._install_jax_backend(env_name, repo_path, commit_year)

    def _parse_tensorflow_version_from_setup(self, repo_path: Path) -> Optional[str]:
        setup_py = repo_path / 'setup.py'

        if not setup_py.exists():
            return None

        try:
            with open(setup_py, 'r', encoding='utf-8') as f:
                content = f.read()

            import re

            pattern1 = r'["\']tensorflow[^"\']*["\']'
            matches = re.findall(pattern1, content)

            if matches:
                tf_spec = matches[0].strip('"\'')
                print(f"    [Parse] Found in setup.py: {tf_spec}")
                return tf_spec

            pyproject_toml = repo_path / 'pyproject.toml'
            if pyproject_toml.exists():
                with open(pyproject_toml, 'r', encoding='utf-8') as f:
                    toml_content = f.read()

                tf_matches = re.findall(r'["\']tensorflow[^"\']*["\']', toml_content)
                if tf_matches:
                    tf_spec = tf_matches[0].strip('"\'')
                    print(f"    [Parse] Found in pyproject.toml: {tf_spec}")
                    return tf_spec

        except Exception as e:
            print(f"    [Warning] Failed to parse setup.py: {e}")

        return None

    def _infer_tensorflow_version(self, repo_path: Path, commit_year: Optional[int] = None) -> Optional[str]:
        keras_version = self._get_keras_version_from_code(repo_path)

        if commit_year and commit_year >= 2025:
            print(f"    [Inference] 2025-era code detected; using TensorFlow 2.16+")
            return "tensorflow>=2.16,<2.18"

        elif commit_year and commit_year >= 2024:
            print(f"    [Inference] 2024-era code detected; using TensorFlow 2.15+")
            return "tensorflow>=2.15,<2.17"

        elif commit_year and commit_year >= 2023:
            print(f"    [Inference] 2023-era code detected; using TensorFlow 2.12-2.14")
            return "tensorflow>=2.12,<2.15"

        elif keras_version:
            major_version = int(keras_version.split('.')[0])

            if major_version >= 3:
                print(f"    [Inference] Keras {keras_version} -> TensorFlow 2.15+")
                return "tensorflow>=2.15,<2.18"
            else:
                # Keras 2.x
                print(f"    [Inference] Keras {keras_version} -> TensorFlow 2.12+")
                return "tensorflow>=2.12,<2.16"

        else:
            print(f"    [Inference] Defaulting to TensorFlow 2.16+")
            return "tensorflow>=2.16,<2.18"

    def _install_jax_backend(self, env_name: str, repo_path: Path, commit_year: Optional[int] = None):
        print(f"    [Keras Backend] Installing the JAX backend...")

        env_prefix = self._get_conda_prefix(env_name)
        pip_path = Path(env_prefix) / 'bin' / 'pip'
        python_path = Path(env_prefix) / 'bin' / 'python'

        jax_version = self._parse_jax_version_from_requirements(repo_path)

        if not jax_version:
            if commit_year and commit_year >= 2025:
                jax_version = "jax[cpu]==0.5.0"
            else:
                jax_version = "jax[cpu]>=0.4.0,<0.6.0"
            print(f"    [Inference] Using default JAX version: {jax_version}")

        try:
            print(f"    [Keras Backend] Installing {jax_version}...")
            result = subprocess.run(
                [str(pip_path), 'install', jax_version, '-q'],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                print(f"    ✓ JAX installation completed")

                verify_result = subprocess.run(
                    [str(python_path), '-c', 'import jax; print(f"JAX {jax.__version__}")'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if verify_result.returncode == 0:
                    print(f"    ✓ JAX validation succeeded: {verify_result.stdout.strip()}")
                else:
                    print(f"    ⚠ JAX validation failed: {verify_result.stderr}")
            else:
                print(f"    ⚠ JAX installation failed: {result.stderr}")
                print(f"    [Note] JAX backend tests may not run")

        except Exception as e:
            print(f"    ⚠ JAX installation error: {e}")

    def _parse_jax_version_from_requirements(self, repo_path: Path) -> Optional[str]:
        requirements_file = repo_path / 'requirements.txt'

        if not requirements_file.exists():
            return None

        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                content = f.read()

            import re
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('jax'):
                    if '#' in line:
                        line = line[:line.index('#')].strip()
                    if line:
                        print(f"    [Parse] Found in requirements.txt: {line}")
                        return line

        except Exception as e:
            print(f"    [Warning] Failed to parse requirements.txt: {e}")

        return None

    def _get_keras_version_from_code(self, repo_path: Path) -> Optional[str]:
        version_files = [
            repo_path / 'keras' / '__init__.py',
            repo_path / 'keras' / 'src' / 'version.py',
            repo_path / 'keras' / '_version.py',
            repo_path / 'keras' / 'version.py',
        ]

        import re
        version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'

        for version_file in version_files:
            if version_file.exists():
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    match = re.search(version_pattern, content)
                    if match:
                        version = match.group(1)
                        print(f"    [Parse] Keras version: {version}")
                        return version
                except Exception:
                    continue

        return None

    def _install_code_only(self, env_name: str, project_path: Path, commit: str):
        print(f"    Installing code for commit {commit[:8]} (following the best-practice flow)")

        env_prefix = self._get_conda_prefix(env_name)
        pip_path = Path(env_prefix) / 'bin' / 'pip'
        python_path = Path(env_prefix) / 'bin' / 'python'

        env_work_dir = self.work_dir / env_name
        env_work_dir.mkdir(parents=True, exist_ok=True)
        repo_path = env_work_dir / 'repo'

        if repo_path.exists():
            shutil.rmtree(repo_path)

        try:
            clone_result = subprocess.run(
                ['git', 'clone', str(project_path), str(repo_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if clone_result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {clone_result.stderr}")

            source_gitmodules = project_path / '.gitmodules'
            if source_gitmodules.exists():
                shutil.copy2(source_gitmodules, repo_path / '.gitmodules')

                source_git_modules = project_path / '.git' / 'modules'
                if source_git_modules.exists():
                    target_git_modules = repo_path / '.git' / 'modules'
                    shutil.copytree(source_git_modules, target_git_modules, dirs_exist_ok=True)

                try:
                    with open(source_gitmodules, 'r') as f:
                        content = f.read()
                        import re
                        paths = re.findall(r'path\s*=\s*(.+)', content)
                        for submod_path in paths:
                            submod_path = submod_path.strip()
                            source_submod = project_path / submod_path
                            target_submod = repo_path / submod_path
                            if source_submod.exists():
                                target_submod.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copytree(source_submod, target_submod, dirs_exist_ok=True)
                                print(f"    ✓ Copied submodule: {submod_path}")
                except Exception as e:
                    print(f"    ⚠ Submodule copy failed (non-fatal): {e}")

            checkout_result = subprocess.run(
                ['git', 'checkout', commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if checkout_result.returncode != 0:
                raise RuntimeError(f"Checkout failed: {checkout_result.stderr}")

            commit_year = None
            try:
                year_result = subprocess.run(
                    ['git', 'show', '-s', '--format=%ci', commit],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if year_result.returncode == 0:
                    commit_year = int(year_result.stdout.strip()[:4])
                    print(f"    [Analysis] Commit year: {commit_year}")
            except Exception:
                pass

            project_name = project_path.name

            install_dir = repo_path
            if project_name in self.python_package_subdir:
                subdir = self.python_package_subdir[project_name]
                install_dir = repo_path / subdir
                print(f"    [Note] Python package lives in subdirectory: {subdir}")

            self._install_build_dependencies(env_name, commit_year)

            self._run_source_generation_scripts(repo_path, project_name, python_path)

            has_cython = bool(list(repo_path.glob('**/*.pyx')))
            needs_compilation = project_name in ['pandas', 'scipy', 'numpy', 'aiohttp'] or has_cython

            if needs_compilation:
                print(f"    [Build] Detected {project_name} (C/C++ dependencies); using strict mode...")

                print(f"    [Uninstall] Removing prebuilt {project_name}...")
                subprocess.run(
                    [str(pip_path), 'uninstall', project_name, '-y', '-q'],
                    capture_output=True,
                    timeout=60
                )

                if project_name in ['pandas', 'scipy'] and commit_year:
                    self._adjust_numpy_for_compilation(env_name, project_name, commit_year)

                print(f"    [Build] Installing (keeping --no-deps)...")
                install_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--no-build-isolation',
                    '--no-deps',
                    '--force-reinstall',
                    '-q',
                ]
                fallback_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--no-deps',
                    '--force-reinstall',
                    '-q',
                ]
                install_timeout = 900
            else:
                print(f"    [Pure Python] Installing (allowing automatic dependency resolution)...")
                install_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--no-build-isolation',
                    '--force-reinstall',
                    '-q',
                ]
                fallback_cmd = [
                    str(pip_path),
                    'install',
                    '.',
                    '--force-reinstall',
                    '-q',
                ]
                install_timeout = 600

            install_result = subprocess.run(
                install_cmd,
                cwd=install_dir,
                capture_output=True,
                text=True,
                timeout=install_timeout
            )

            if install_result.returncode != 0:
                print(f"    [Warning] No-isolation install failed; retrying with a standard install...")
                install_result = subprocess.run(
                    fallback_cmd,
                    cwd=install_dir,
                    capture_output=True,
                    text=True,
                    timeout=install_timeout
                )

                if install_result.returncode != 0:
                    if needs_compilation:
                        print(f"    [Error] Compiled installation failed")
                        print(f"    stderr: {install_result.stderr[-500:]}")
                        raise RuntimeError(f"Compiled installation failed")
                    else:
                        print(f"    [Error] Code installation failed: {install_result.stderr}")
                        raise RuntimeError(f"Code installation failed")

            if needs_compilation:
                print(f"    ✓ Code compilation and installation completed")
            else:
                print(f"    ✓ Code installation completed (dependencies resolved)")

            if project_name.lower() == 'keras':
                self._install_keras_backend(env_name, repo_path, commit_year)

            self._verify_installation(python_path, project_name)

            compile_projects = {'pandas', 'scipy', 'numpy', 'aiohttp'}
            if project_name.lower() in compile_projects:
                self._verify_compilation(python_path, project_name)

        except Exception as e:
            print(f"    [Error] Installation process error: {e}")
            raise

    def _get_conda_prefix(self, env_name: str) -> str:
        """Return the path of a conda environment."""
        prefix_path = self._get_env_prefix_path(env_name)
        python_path = prefix_path / 'bin' / 'python'
        if python_path.exists():
            return str(prefix_path)
        if prefix_path.exists():
            return str(prefix_path)

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

    def get_env_prefix(self, env_name: str) -> str:
        """Public helper that returns the conda prefix path."""
        return self._get_conda_prefix(env_name)

    def run_test_in_env(self, env_name: str, test_code: str) -> Dict[str, Any]:
        if not self._env_exists(env_name):
            error_msg = f"Conda environment not found: {env_name}"
            print(f"  ✗ {error_msg}")

            try:
                result = subprocess.run(
                    ['conda', 'env', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  Available environments:")
                    for line in result.stdout.split('\n'):
                        if line.strip() and not line.startswith('#'):
                            print(f"    - {line.strip().split()[0]}")
            except Exception:
                pass

            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': error_msg,
                'error': error_msg,
                'debug_output': ''
            }

        cwd_path = self._get_env_repo_dir(env_name)
        if not cwd_path.exists():
            print(f"  [Warning] Source checkout not found: {cwd_path}; running in the system temp directory")
            cwd_path = None

        enhanced_test_code = self._enhance_test_code(test_code, env_name)

        target_dir = cwd_path if cwd_path else Path(tempfile.gettempdir())
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=str(target_dir), delete=False) as f:
            f.write(enhanced_test_code)
            test_file = f.name

        try:
            conda_prefix = self._get_conda_prefix(env_name)
            python_path = Path(conda_prefix) / 'bin' / 'python'
            pytest_path = Path(conda_prefix) / 'bin' / 'pytest'

            # -v: verbose
            clean_env = os.environ.copy()
            clean_env.pop('PYTHONPATH', None)
            clean_env['PATH'] = f"{conda_prefix}/bin:" + clean_env.get('PATH', '')

            result = subprocess.run(
                [str(pytest_path), test_file, '-v', '-s', '--tb=short', '--capture=no'],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(cwd_path) if cwd_path else None,
                env=clean_env
            )
            if result.returncode != 0:
                print(f"\n{'!'*20} Pytest Execution Failed {'!'*20}")
                print(f"Environment: {env_name}")
                
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"❌ Error Output:\n{error_msg}")
                
                print(f"📜 Generated Code (First 20 lines):\n")
                print('\n'.join(enhanced_test_code.split('\n')[:20]))
                print(f"{'!'*60}\n")

            # Pytest outputs errors to stdout, not stderr
            error_output = result.stdout if result.returncode != 0 else ''
            if not error_output and result.stderr:
                error_output = result.stderr

            debug_output = self._extract_debug_output(result.stdout)

            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'error': error_output,
                'debug_output': debug_output
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test execution timeout (60s)',
                'error': 'Test execution timeout',
                'debug_output': ''
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'error': str(e),
                'debug_output': ''
            }
        finally:
            Path(test_file).unlink(missing_ok=True)

    def _enhance_test_code(self, test_code: str, env_name: str) -> str:
        enhanced_code = f'''
import sys
print("\\n" + "="*70)
print(f"[Test Environment] {{repr({repr(env_name)})}}")
print(f"[Python Version] {{sys.version.split()[0]}}")
print("="*70 + "\\n")

{test_code}

print("\\n" + "="*70)
print("[Test Execution] Completed")
print("="*70)
'''
        return enhanced_code

    def _extract_debug_output(self, stdout: str) -> str:
        if not stdout:
            return ""

        debug_lines = []
        in_test_output = False

        for line in stdout.split('\n'):
            if '[Test Environment]' in line or '====' in line and 'Test Environment' in stdout:
                in_test_output = True

            if in_test_output:
                if not any(marker in line for marker in [
                    'test session starts',
                    'platform linux',
                    'cachedir:',
                    'rootdir:',
                    'collected',
                    'PASSED',
                    'FAILED',
                    '::test_',
                    'short test summary'
                ]):
                    debug_lines.append(line)

            if '[Test Execution] Completed' in line:
                in_test_output = False

        return '\n'.join(debug_lines).strip()

    def _verify_env_commit(self, env_name: str, expected_commit: str) -> bool:
        try:
            env_work_dir = self.work_dir / env_name
            commit_file = env_work_dir / 'commit_hash.txt'

            if not commit_file.exists():
                return False

            saved_commit = commit_file.read_text().strip()
            if saved_commit != expected_commit:
                return False

            parts = env_name.rsplit('_pr', 1)
            if len(parts) != 2:
                return False

            pr_index = env_name.rfind('_pr')
            project_name = env_name[:pr_index]

            conda_prefix = self._get_env_prefix_path(env_name)
            python_path = conda_prefix / "bin" / "python"

            if not python_path.exists():
                return False

            import_name = VERIFY_IMPORT_NAME_MAPPING.get(project_name, project_name.replace('-', '_'))

            result = subprocess.run(
                [str(python_path), '-c', f'import {import_name}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.returncode == 0

        except Exception:
            return False

    def _save_env_commit(self, env_name: str, commit: str):
        try:
            env_work_dir = self.work_dir / env_name
            env_work_dir.mkdir(parents=True, exist_ok=True)

            commit_file = env_work_dir / 'commit_hash.txt'
            commit_file.write_text(commit)

        except Exception as e:
            print(f"    [Warning] Failed to save commit metadata: {e}")

    def cleanup_env(self, env_name: str):
        """Clean a conda environment and its related work directory."""
        try:
            prefix_path = self._get_env_prefix_path(env_name)
            if prefix_path.exists():
                remove_cmd = ['conda', 'env', 'remove', '-p', str(prefix_path), '-y', '-q']
            else:
                remove_cmd = ['conda', 'env', 'remove', '-n', env_name, '-y', '-q']
            subprocess.run(
                remove_cmd,
                timeout=60,
                capture_output=True
            )

            if prefix_path.exists():
                shutil.rmtree(prefix_path)

            env_work_dir = self.work_dir / env_name
            if env_work_dir.exists():
                shutil.rmtree(env_work_dir)

            print(f"  ✓ Cleaned environment: {env_name}")
        except Exception as e:
            print(f"  [Warning] Failed to clean environment {env_name}: {e}")

    def cleanup_pr_environments(self, pre_env_name: str, post_env_name: str):
        """Clean all environments associated with a PR."""
        print(f"\n[Environment Cleanup]")
        self.cleanup_env(pre_env_name)
        self.cleanup_env(post_env_name)

    def _ensure_test_tooling(self, env_name: str):
        try:
            env_prefix = self._get_conda_prefix(env_name)
        except RuntimeError as exc:
            print(f"    [Warning] Unable to check test tooling ({exc})")
            return

        python_path = Path(env_prefix) / 'bin' / 'python'
        pip_path = Path(env_prefix) / 'bin' / 'pip'

        requirements = {
            'pytest': 'pytest>=7.4',
            'coverage': 'coverage',
        }

        for module_name, package_spec in requirements.items():
            check_cmd = [
                str(python_path),
                '-c',
                (
                    "import importlib, sys\n"
                    f"sys.exit(0) if importlib.util.find_spec('{module_name}') else sys.exit(1)"
                ),
            ]
            try:
                check = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            except Exception as exc:
                print(f"    [Warning] Failed to check {module_name}: {exc}")
                check = None

            if check is not None and check.returncode == 0:
                print(f"    ✓ {module_name} is already installed in {env_name}")
                continue

            print(f"    [Install] {package_spec} -> {env_name}")
            max_retries = 3
            install_success = False

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"    [Retry] Attempt {attempt + 1} to install {package_spec}...")
                        time.sleep(2)

                    install = subprocess.run(
                        [str(pip_path), 'install', package_spec, '-q'],
                        capture_output=True,
                        text=True,
                        timeout=180
                    )

                    if install.returncode == 0:
                        print(f"    ✓ {package_spec} installed successfully")
                        install_success = True
                        break
                    else:
                        error_msg = install.stderr.strip() if install.stderr else "unknown error"
                        print(f"    [Warning] Failed to install {package_spec} (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")

                except subprocess.TimeoutExpired:
                    print(f"    [Warning] Installing {package_spec} timed out (attempt {attempt + 1}/{max_retries})")
                except Exception as exc:
                    print(f"    [Warning] Installing {package_spec} raised an error (attempt {attempt + 1}/{max_retries}): {exc}")

            if not install_success:
                print(f"    [Error] Failed to install {package_spec} after {max_retries} retries")

    def _install_project_extras(self, project: str, env_name: str):
        extras = self.project_extra_dependencies.get(project)
        if not extras:
            return

        try:
            env_prefix = self._get_conda_prefix(env_name)
        except RuntimeError as exc:
            print(f"    [Warning] Unable to install extra dependencies ({exc})")
            return

        pip_path = Path(env_prefix) / 'bin' / 'pip'
        cmd = [str(pip_path), 'install', '-q'] + extras
        print(f"    [Dependencies] Installing extra dependencies for project {project}: {' '.join(extras)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"    [Warning] Failed to install extra dependencies: {result.stderr.strip()}")
        except Exception as exc:
            print(f"    [Warning] Error while installing extra dependencies: {exc}")


