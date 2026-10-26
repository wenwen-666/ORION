"""Prompt building blocks for the regression classifier."""

FIXED_SYSTEM_RULES = """You are a regression detector.
Determine if a code change causes functional or performance regression based ONLY on objective execution data.

Strict Rules (must follow):
1. If execution status changes from OK to EXCEPTION/TIMEOUT/FAILED -> regression.
2. If exception behavior changes (new exception, OR type changed, OR message changed significantly) -> regression.
3. If return value/stdout changes unexpectedly -> regression.
4. If execution time slowdown >= 2x (ratio >= 2.0) -> performance regression.
5. New function (no prior execution) -> NOT regression (clean).
6. Infrastructure error -> unknown (cannot determine regression).

IMPORTANT - Environmental noise to IGNORE (these are NOT regressions):
- File paths containing conda env names (e.g. /conda_envs/pkg_pre/ vs /conda_envs/pkg_post/) - these differ between pre/post environments by design.
- Temporary file paths (e.g. /tmp/tmpXXXXXX) - these are randomly generated each run.
- Memory addresses (e.g. 0x7f...) - these vary between runs.
- Execution time differences < 2x - normal variance.
If the ONLY stdout/output difference is one of the above, answer Q4=no."""

SYSTEM_PROMPT_BODY = """Task:
Apply the strict rules above before any softer judgment.
Use only objective execution evidence when deciding Q1-Q5 and the final regression label.
Return ONLY valid JSON with Q1-Q5, regression_label, confidence, and reason."""

SYSTEM_PROMPT = FIXED_SYSTEM_RULES + "\n\n" + SYSTEM_PROMPT_BODY

QUESTION_BLOCK = """=== YOUR TASK ===
Answer these 5 OBJECTIVE yes/no questions using only the execution evidence above.

Q1: Is there any objective behavior difference between BEFORE and AFTER (status, exception, return value/stdout, or execution time)? (yes/no)
Q2: Did execution status regress from OK to EXCEPTION, TIMEOUT, or FAILED? (yes/no)
Q3: Did exception behavior change? Answer yes if a new exception appeared, the exception type changed, or the exception message changed significantly. (yes/no)
Q4: Did return value or stdout change unexpectedly after ignoring conda-env paths, temporary file paths, and memory addresses? (yes/no)
Q5: Did execution time slow down by at least 2x (ratio >= 2.0)? (yes/no)

Output JSON (strict format, NO extra text):
{
  "Q1": "yes/no",
  "Q2": "yes/no",
  "Q3": "yes/no",
  "Q4": "yes/no",
  "Q5": "yes/no",
  "regression_label": "regression/clean/unknown",
  "confidence": 0.0-1.0,
  "reason": "objective justification based on execution data"
}

Rule for regression_label: If Q1=yes AND any Q2-Q5=yes -> regression; else -> clean; infra error -> unknown."""
