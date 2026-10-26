"""Centralized local configuration for ORION.

This file is intentionally lightweight for the public repository version.
ORION does not ship benchmark-specific default mappings in code. You must
provide a local JSON config file via `ORION_CONFIG_JSON` before running the
pipeline.

Example environment overrides:
  export ORION_LLM_API_KEY="<your_api_key>"
  export ORION_LLM_MODEL="<your_model_name>"
  export ORION_OPENAI_BASE_URL="<your_llm_base_url>"
  export ORION_PROJECT_BASE="/path/to/ORION/project"
  export ORION_CONFIG_JSON="/path/to/config.override.json"
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_OVERRIDE_JSON = os.environ.get("ORION_CONFIG_JSON")

# Example: export ORION_OPENAI_BASE_URL="<your_llm_base_url>"
# Leave this unset if your client should use its provider default base URL.
OPENAI_COMPATIBLE_BASE_URL = os.environ.get("ORION_OPENAI_BASE_URL")

# Example: export ORION_LLM_MODEL="<your_model_name>"
DEFAULT_LLM_MODEL = os.environ.get("ORION_LLM_MODEL", "<your_model_name>")


def resolve_llm_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the API key used by LLM-backed stages."""
    return explicit_api_key or os.environ.get("ORION_LLM_API_KEY")


def build_openai_client_kwargs(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for an OpenAI-compatible client."""
    kwargs: Dict[str, Any] = {}
    resolved_api_key = resolve_llm_api_key(api_key)
    resolved_base_url = base_url if base_url is not None else OPENAI_COMPATIBLE_BASE_URL

    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url

    return kwargs


def _load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"ORION config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"ORION config file must contain a JSON object at the top level: {path}")

    return data


def _load_user_config() -> Dict[str, Any]:
    """Load the required user-provided JSON config."""
    if not CONFIG_OVERRIDE_JSON:
        raise RuntimeError(
            "ORION requires a user config file. Set ORION_CONFIG_JSON to a JSON file "
            "based on config.override.example.json before running the pipeline."
        )
    return _load_json_object(Path(CONFIG_OVERRIDE_JSON))


CONFIG_VALUES = _load_user_config()


def _resolve_value(name: str) -> Any:
    """Resolve a required setting from the user config."""
    if name not in CONFIG_VALUES:
        raise KeyError(f"Missing ORION config key: {name}")
    return CONFIG_VALUES[name]


def _resolve_dict(name: str) -> Dict[str, Any]:
    """Resolve a dict setting."""
    value = _resolve_value(name)
    if not isinstance(value, dict):
        raise ValueError(f"ORION config key '{name}' must be a JSON object")
    return dict(value)


def _resolve_str_list(name: str) -> Tuple[str, ...]:
    """Resolve a string-list setting as a tuple."""
    value = _resolve_value(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"ORION config key '{name}' must be a JSON array of strings")
    return tuple(value)


def _resolve_str_set(name: str) -> Set[str]:
    """Resolve a string-list setting as a set."""
    value = _resolve_value(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"ORION config key '{name}' must be a JSON array of strings")
    return set(value)


PR_METADATA_DIR_BY_TYPE = _resolve_dict("PR_METADATA_DIR_BY_TYPE")
OUTPUT_SUBDIR_BY_PR_TYPE = _resolve_dict("OUTPUT_SUBDIR_BY_PR_TYPE")
PR_METADATA_FILENAME_PATTERNS: Tuple[str, ...] = _resolve_str_list("PR_METADATA_FILENAME_PATTERNS")
PROJECT_PATH_MAPPING = _resolve_dict("PROJECT_PATH_MAPPING")
PYTHON_PACKAGE_SUBDIR = _resolve_dict("PYTHON_PACKAGE_SUBDIR")
IMPORT_NAME_MAPPING = _resolve_dict("IMPORT_NAME_MAPPING")
VERIFY_IMPORT_NAME_MAPPING = _resolve_dict("VERIFY_IMPORT_NAME_MAPPING")
PROJECT_EXTRA_DEPENDENCIES = _resolve_dict("PROJECT_EXTRA_DEPENDENCIES")
DEFAULT_PYTHON_VERSION_BY_PROJECT = _resolve_dict("DEFAULT_PYTHON_VERSION_BY_PROJECT")
SKIP_IMPORT_VALIDATION_PROJECTS = _resolve_str_set("SKIP_IMPORT_VALIDATION_PROJECTS")
BASE_ENV_TEMPLATES = _resolve_dict("BASE_ENV_TEMPLATES")


def resolve_project_base(base_dir: Path) -> Path:
    """Return the repository-checkout root for the current machine."""
    return Path(os.environ.get("ORION_PROJECT_BASE", str(base_dir / "project")))


def resolve_project_checkout(project_base: Path, repo: str) -> Path:
    """Return the checkout path for a repository key."""
    return project_base / PROJECT_PATH_MAPPING.get(repo, repo)


def resolve_project_checkout_from_base_dir(base_dir: Path, repo: str) -> Path:
    """Resolve a repository checkout path from the configured base directory."""
    return resolve_project_checkout(resolve_project_base(base_dir), repo)


def iter_pr_metadata_candidates(base_dir: Path, repo: str, pr: int):
    """Yield candidate metadata paths in priority order."""
    for pr_type in ("positive", "negative", "new202506"):
        root = base_dir / PR_METADATA_DIR_BY_TYPE[pr_type] / repo
        for pattern in PR_METADATA_FILENAME_PATTERNS:
            yield pr_type, root / pattern.format(pr=pr)


def resolve_pr_metadata(base_dir: Path, repo: str, pr: int):
    """Resolve the metadata path and PR type for a given repo/PR pair."""
    fallback = base_dir / PR_METADATA_DIR_BY_TYPE["positive"] / repo / PR_METADATA_FILENAME_PATTERNS[0].format(pr=pr)
    for pr_type, path in iter_pr_metadata_candidates(base_dir, repo, pr):
        if path.exists():
            return path, pr_type
    return fallback, "positive"


def metadata_dir_for_type(base_dir: Path, pr_type: str) -> Path:
    """Return the metadata directory root for a PR type."""
    return base_dir / PR_METADATA_DIR_BY_TYPE[pr_type]


def output_base_for_type(base_dir: Path, pr_type: str) -> Path:
    """Return the output root for a PR type."""
    subdir = OUTPUT_SUBDIR_BY_PR_TYPE[pr_type]
    return base_dir if not subdir else base_dir / subdir


def primary_repo_metadata_dir(base_dir: Path) -> Path:
    """Return the default repository listing root."""
    return base_dir / PR_METADATA_DIR_BY_TYPE["positive"]
