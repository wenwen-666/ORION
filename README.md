# ORION

## Overview

ORION is a PR-level regression analysis pipeline for Python pull requests.

This repository also includes our benchmark dataset, **RegBench-Py**, under `dataset/`.
The dataset is organized into `dataset/positive_pr/` and `dataset/negative_pr/`, where each PR is stored as a JSON file with the associated metadata needed by the pipeline.

Given a repository name and a PR number, it runs the following stages in order:

- **Stage 0**: Build the impact region.
- **Stage 1**: Extract PR intent (LLM).
- **Stage 2**: Generate function oracles (LLM).
- **Stage 3**: Build the target function view.
- **Stage 4**: Generate probes or test specs (LLM).
- **Stage 5**: Generate the probe runner.
- **Stage 6**: Execute probes in pre-commit and post-commit environments, then summarize behavior differences.
- **Stage 7**: Run intendedness classification (LLM).

## Repository Layout

The repository already includes the benchmark dataset metadata, but runtime assets such as checked-out projects and conda environments still need to be prepared locally.
The typical layout is:

```text
ORION/
  dataset/
    positive_pr/
    negative_pr/
  project/
  conda_envs/
  conda_env_workspace/
```

- `dataset/` contains **RegBench-Py**, the PR metadata used by ORION.
- `dataset/positive_pr/` and `dataset/negative_pr/` store the benchmark PR JSON files.
- `project/` stores the checked-out target repositories.
- `conda_envs/` stores reusable conda environments created during Stage 6.
- `conda_env_workspace/` stores temporary working directories used during environment setup.

## Prerequisites

Install the required Python packages used by ORION itself:

```bash
cd /path/to/ORION
python3 -m pip install -r requirements.txt
```

Make sure `conda` and `git` are installed separately. They are external system prerequisites and are not installed from `requirements.txt`.

Set the API key before running any LLM-based stage (Stage 1, 2, 4, or 7):

```bash
export ORION_LLM_API_KEY="<your_api_key>"
export ORION_LLM_MODEL="<your_model_name>"
```

If your provider requires a custom OpenAI-compatible endpoint, set it explicitly as well:

```bash
export ORION_OPENAI_BASE_URL="<your_llm_base_url>"
```

Before running the pipeline, create your own JSON config file from `config.override.example.json`, fill in the repository-specific values, and point ORION to it:

```bash
export ORION_CONFIG_JSON="/path/to/config.override.json"
```

When using the bundled **RegBench-Py** dataset, set `PR_METADATA_DIR_BY_TYPE` in your config to the dataset directories in this repository, for example:

```json
{
  "PR_METADATA_DIR_BY_TYPE": {
    "positive": "/path/to/ORION/dataset/positive_pr",
    "negative": "/path/to/ORION/dataset/negative_pr"
  }
}
```

The repository includes the dataset itself, but project path mappings, import mappings, environment templates, and other execution-specific settings still need to be filled in by the user.

## Main Command

Run a single PR:

```bash
cd /path/to/ORION
python3 pipeline.py --repo astroid --pr 2459
```

You can also run it with the absolute path:

```bash
python3 /path/to/ORION/pipeline.py --repo astroid --pr 2459
```

## Common Usage

Show help:

```bash
cd /path/to/ORION
python3 pipeline.py --help
```

Run all PRs for one repository:

```bash
cd /path/to/ORION
python3 pipeline.py --repo astroid
```

Run all repositories:

```bash
cd /path/to/ORION
python3 pipeline.py --all
```

Force a full re-run instead of resuming from completed stages:

```bash
cd /path/to/ORION
python3 pipeline.py --repo astroid --pr 2459 --force
```

## Outputs

The main output locations are:

- `logs/`
- `impact_region/output/`
- `intent_extraction/output/`
- `function_oracle_generation/output/`
- `target_function_view/output/`
- `test_probe_generation/output/`
- `test_executor/output/`
- `intendedness/output/`

The final regression decision in Stage 7 is based on the intendedness classifier only.

## Notes

- The pipeline supports resume behavior by default and skips completed stages.
- Stage 1, 2, 4, and 7 depend on LLM/API access.
- Stage 6 depends on conda environment creation or reuse.
- Stage 7 uses a fixed intendedness prompt configuration.
