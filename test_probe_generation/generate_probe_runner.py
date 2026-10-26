#!/usr/bin/env python3
"""Generate the pytest runner used to execute probe tests for one PR.

The generated runner is path-agnostic: it relies on the working directory and
environment variables prepared by the test executor so each commit is tested in
its own isolated environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_BASE_DIR = os.environ.get("ORION_BASE_DIR", str(Path(__file__).resolve().parents[1]))


def generate_pytest_content(json_path: str) -> str:
    safe_json_path = json_path.replace("\\", "\\\\")

    path_setup_code = '''
import sys
import os

cwd = os.getcwd()
# 1. Repo root
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# 2. src layout support
src_path = os.path.join(cwd, "src")
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"DEBUG: Probe Runner CWD: {cwd}")
print(f"DEBUG: sys.path[0:2]: {sys.path[:2]}")
'''

    return f'''import contextlib
import importlib
import io
import json
import os
import warnings
import traceback
import pytest

{path_setup_code}

OUTPUT_DIR = os.environ.get("PR_TEST_OUTPUT_DIR", "outputs")
PROBES_JSON_PATH = r"{safe_json_path}"

try:
    with open(PROBES_JSON_PATH, "r") as fh:
        data = json.load(fh)
        PROBES = data.get("probes") or data.get("specs") or []
except Exception as exc:
    print(f"FATAL: Cannot load probes: {{exc}}")
    PROBES = []

ENV_HINTS = {{
    "backend": "KERAS_BACKEND",
    "nnx_enabled": "KERAS_NNX_ENABLED",
}}

def _write_result(probe_id, payload):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{{probe_id}}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)

def _prepare_preconditions(preconditions):
    applied = {{}}
    env_patch = {{}}
    for key, value in (preconditions or {{}}).items():
        env_key = ENV_HINTS.get(key)
        if env_key:
            env_value = "1" if isinstance(value, bool) and value else ("0" if isinstance(value, bool) else str(value))
            env_patch[env_key] = env_value
            applied[key] = env_value
        else:
            if key.isupper(): env_patch[key] = str(value)
            applied[key] = value
    return applied, env_patch

@contextlib.contextmanager
def _env_guard(preconditions, result):
    applied, env_patch = _prepare_preconditions(preconditions)
    old_env = {{}}
    try:
        for env_key, env_value in env_patch.items():
            old_env[env_key] = os.environ.get(env_key)
            os.environ[env_key] = env_value
        if applied: result["preconditions_applied"] = applied
        yield
    finally:
        for env_key, old_val in old_env.items():
            if old_val is None: os.environ.pop(env_key, None)
            else: os.environ[env_key] = old_val

def _resolve_callable(target_path):
    if not target_path or not isinstance(target_path, str): raise ValueError("Invalid target")
    parts = target_path.split(".")
    for idx in range(len(parts), 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            module = importlib.import_module(module_name)
            obj = module
            for attr in parts[idx:]: obj = getattr(obj, attr)
            return obj
        except (ModuleNotFoundError, AttributeError): continue
    raise ImportError(f"Cannot resolve: {{target_path}}")

def _canonicalize_value(value):
    if value is None: return None
    if isinstance(value, (bool, int, float, str)): return value
    if isinstance(value, (list, tuple, set)):
        return {{"type": type(value).__name__, "length": len(value), "sample": str(list(value)[:5])}}
    if isinstance(value, dict):
        return {{"type": "dict", "keys": list(value.keys()), "length": len(value)}}
    if hasattr(value, "__dict__"):
        try:
            simple_state = {{}}
            for k, v in vars(value).items():
                if not k.startswith("_") and isinstance(v, (bool, int, float, str, type(None))):
                    simple_state[k] = v
            return {{"type": type(value).__name__, "state": simple_state}}
        except Exception:
            pass
    if hasattr(value, "shape"): return {{"type": type(value).__name__, "shape": str(value.shape)}}
    return {{"type": type(value).__name__, "repr": str(value)[:100]}}

def _capture_observations(result, observations, ret=None, exc=None):
    obs_payload = {{}}
    for key in observations:
        if key == "exception_type": obs_payload[key] = exc.__class__.__name__ if exc else None
        elif key == "exception_message": obs_payload[key] = str(exc) if exc else None
        elif key == "return_value": obs_payload[key] = _canonicalize_value(ret) if exc is None else None
    if obs_payload: result["observations"] = obs_payload

def _classify_code_error(exc):
    msg = str(exc).lower()
    if isinstance(exc, ImportError) or "module" in msg: return "import_error"
    if isinstance(exc, SyntaxError): return "syntax_error"
    if isinstance(exc, (TypeError, AttributeError, ValueError)): return "call_error"
    if isinstance(exc, (MemoryError, RecursionError)): return "resource_error"
    if isinstance(exc, (OSError, IOError)): return "io_error"
    if isinstance(exc, SystemExit): return "system_exit"
    if isinstance(exc, KeyboardInterrupt): return "interrupted"
    return "runtime_error"

def _run_code_probe(probe, code):
    probe_id = probe.get("probe_id") or "unknown"
    result = {{
        "probe_id": probe_id,
        "oracle_ref": probe.get("oracle_ref"),
        "status": "unknown",
        "error": None,
        "error_kind": None,
        "stdout": "",
        "stderr": "",
        "warnings": [],
        "execution_time": None
    }}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Timeout handler for individual probe execution
    import signal
    import time

    def timeout_handler(signum, frame):
        raise TimeoutError("Probe execution exceeded 30 seconds")

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        # Set timeout for probe execution (30 seconds per probe)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        start_time = time.time()
        caught_exception = None

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, {{"__builtins__": __builtins__}})
            result["status"] = "success"
        except BaseException as exc:  # Catch SystemExit, KeyboardInterrupt, etc.
            caught_exception = exc
            result["status"] = "exception"
            result["error"] = f"{{exc.__class__.__name__}}: {{exc}}"
            result["error_kind"] = _classify_code_error(exc)
            result["traceback"] = traceback.format_exc()

            # DEBUG: Print code on error to help debug SyntaxErrors
            print(f"\\n[DEBUG FAIL] Probe: {{probe_id}}")
            print("-" * 40)
            lines = code.splitlines()
            for i, line in enumerate(lines):
                prefix = ">> " if (isinstance(exc, SyntaxError) and getattr(exc, 'lineno', -1) == i + 1) else "   "
                print(f"{{prefix}}{{i+1:3}}: {{line}}")
            print("-" * 40)
            print(f"Error: {{exc}}")
        finally:
            # Always cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            # Record execution time
            result["execution_time"] = time.time() - start_time

    if captured_warnings:
        result["warnings"] = [f"{{w.category.__name__}}: {{str(w.message)}}" for w in captured_warnings]
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

    # Capture structured observations if probe defines them
    observations = probe.get("observations", [])
    if observations:
        _capture_observations(result, observations, ret=None, exc=caught_exception)

    # Protect _write_result call
    try:
        _write_result(probe_id, result)
    except Exception as write_exc:
        print(f"\\n[CRITICAL] Failed to write result for {{probe_id}}: {{write_exc}}")
        # Try to write a minimal error file
        try:
            import tempfile
            fallback_path = os.path.join(tempfile.gettempdir(), f"{{probe_id}}_error.json")
            with open(fallback_path, "w") as f:
                json.dump({{"probe_id": probe_id, "status": "write_error", "error": str(write_exc)}}, f)
            print(f"[CRITICAL] Wrote fallback to {{fallback_path}}")
        except:
            pass  # Give up if even fallback fails

def _execute_probe(probe):
    code = probe.get("probe_code") or probe.get("code")
    if code and code.strip():
        _run_code_probe(probe, code)
        return
    _write_result(probe.get("probe_id"), {{"status": "skipped", "error": "No code"}})

if PROBES:
    @pytest.mark.parametrize("probe", PROBES, ids=[p.get("probe_id", str(i)) for i, p in enumerate(PROBES)])
    def test_probe_runner(probe):
        _execute_probe(probe)
else:
    def test_empty(): pytest.skip("No probes")
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    work_root = Path(args.base_dir)
    repo = args.repo
    pr_number = args.pr

    spec_file = None
    for d in ["test_probe_generation", "test_spec_generation"]:
        candidate = work_root / d / "output" / repo / f"pr_{pr_number}_test_specs.json"
        if candidate.exists():
            spec_file = candidate
            break
            
    if not spec_file:
        print("Error: Spec file not found")
        sys.exit(1)

    out_dir = spec_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    content = generate_pytest_content(str(spec_file.resolve()))
    
    test_file = out_dir / f"pr_{pr_number}_tests.py"
    with open(test_file, "w") as f:
        f.write(content)
        
    print(f"Generated runner: {test_file}")

if __name__ == "__main__":
    main()
