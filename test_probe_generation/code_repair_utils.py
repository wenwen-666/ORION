#!/usr/bin/env python3
"""
Shared utilities for static analysis-based code repair.
Version: 17.0 (Enhanced with AST validation and conservative import fixing)
"""

import ast
import re
import textwrap
from typing import Dict, List, Set, Optional

PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield', 'id', 'type', 'print'
}

COMMON_STD_LIBS = {
    'mock': 'from unittest import mock',
    'patch': 'from unittest.mock import patch',
    'MagicMock': 'from unittest.mock import MagicMock',
    'pytest': 'import pytest',
    'json': 'import json',
    'os': 'import os',
    'sys': 'import sys',
    're': 'import re',
    'datetime': 'import datetime',
    'timedelta': 'from datetime import timedelta',
    'dataclass': 'from dataclasses import dataclass',
    'partial': 'from functools import partial',
    'wraps': 'from functools import wraps',
    'Path': 'from pathlib import Path',
}

def parse_valid_symbols(valid_imports_block: str) -> Dict[str, str]:
    """Parse Valid Imports to build a mapping: symbol_name -> full_import_statement."""
    symbol_map: Dict[str, str] = {}
    if not valid_imports_block: return symbol_map

    for line in valid_imports_block.split('\n'):
        line = line.strip()
        if not line or not line.startswith(('import', 'from')): continue

        match_from = re.search(r'from\s+([\w\.]+)\s+import\s+(.+)', line)
        if match_from:
            symbols_str = match_from.group(2)
            for sym in symbols_str.split(','):
                sym = sym.strip()
                if not sym: continue
                if ' as ' in sym:
                    parts = sym.split(' as ')
                    symbol_map[parts[0].strip()] = line
                    symbol_map[parts[1].strip()] = line
                else:
                    symbol_map[sym] = line
            continue

        match_import = re.search(r'import\s+([\w\.]+)(?:\s+as\s+([\w]+))?', line)
        if match_import:
            module_path = match_import.group(1)
            alias = match_import.group(2)
            symbol_map[module_path.split('.')[0]] = line
            symbol_map[module_path] = line
            if alias: symbol_map[alias] = line

    return symbol_map


def detect_used_symbols(code: str) -> Set[str]:
    """Detect all symbols (identifiers) used in the code."""
    used_symbols: Set[str] = set()
    all_tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', code)
    for token in all_tokens:
        if token not in PYTHON_KEYWORDS:
            used_symbols.add(token)
    return used_symbols


def clean_llm_code(code: str) -> str:
    """Extract code from markdown and dedent safely."""
    if not code: return ""
    text = str(code)
    fenced = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced: text = fenced.group(1)
    else:
        generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if generic: text = generic.group(1)
    return textwrap.dedent(text).strip()


def fix_symbol_typos(code: str) -> str:
    """
    Fixes 'missing_' -> 'missing' while protecting keywords like 'class_'.
    """
    if not code: return ""
    
    def replacement(match):
        full_word = match.group(0)
        word_body = match.group(1)
        if word_body in PYTHON_KEYWORDS:
            return full_word
        return word_body

    fixed_lines = []
    for line in code.splitlines():
        if line.strip().startswith(("'", '"')):
            fixed_lines.append(line)
            continue
        new_line = re.sub(r'\b([A-Za-z_][A-Za-z0-9]*)_(?!\w)', replacement, line)
        fixed_lines.append(new_line)
        
    return "\n".join(fixed_lines)


def validate_and_fix_imports(code: str, module_name: str, symbol_map: Dict[str, str]) -> str:
    """
    Conservative import fixer: prefer known symbols, fall back to stdlib hints.
    Only modify imports when we're confident about the fix.
    """
    symbol_map = symbol_map or {}
    new_lines = []
    root_pkg = module_name.split('.')[0] if module_name else ""

    for line in code.splitlines():
        sline = line.strip()
        indent = line[:len(line) - len(line.lstrip())]

        match_from = re.search(r'^from\s+([\w\.]+)\s+import\s+(.+)', sline)
        if match_from:
            current_path = match_from.group(1)
            imported_syms = [sym.strip().split()[0] for sym in match_from.group(2).split(',') if sym.strip()]
            reconstructed: List[str] = []
            for sym in imported_syms:
                if sym in symbol_map:
                    # Use the known-good import from symbol_map
                    reconstructed.append(indent + symbol_map[sym])
                elif sym in COMMON_STD_LIBS:
                    # Use stdlib import
                    reconstructed.append(indent + COMMON_STD_LIBS[sym])
                elif root_pkg and current_path.startswith(root_pkg + ".") and sym.islower() and len(current_path.split('.')) > 2:
                    # CONSERVATIVE: Only downgrade if path is deeply nested (e.g., lib.sub.subsub.module)
                    # AND symbol is lowercase (likely a function/variable, not a class)
                    print(f"    [Auto-Fix] Conservative downgrade: '{sym}' from {current_path} to {root_pkg}")
                    reconstructed.append(indent + f"from {root_pkg} import {sym}")
                elif sym.isupper() and sym not in symbol_map and root_pkg and current_path.startswith(root_pkg + "."):
                    # CHANGED: Don't drop uppercase symbols, but warn and keep original import
                    print(f"    [WARN] Unknown uppercase symbol '{sym}' from {current_path} - keeping original import")
                    reconstructed.append(indent + f"from {current_path} import {sym}")
                else:
                    # Keep original import when unsure
                    reconstructed.append(indent + f"from {current_path} import {sym}")
            if reconstructed:
                new_lines.extend(reconstructed)
                continue

        match_bare_class = re.match(r'^import\s+([A-Z][\w]+)(?:\s|#|$)', sline)
        if match_bare_class:
            target = match_bare_class.group(1)
            if target in symbol_map:
                new_lines.append(indent + symbol_map[target])
                continue
            if module_name:
                fixed_import = f"from {module_name} import {target}"
                print(f"    [Auto-Fix] Converting bare class import: {sline} -> {fixed_import}")
                new_lines.append(indent + fixed_import)
                continue

        match_bare_generic = re.match(r'^import\s+([\w_]+)(?:\s|#|$)', sline)
        if match_bare_generic:
            target = match_bare_generic.group(1)
            if target in symbol_map and "import " in symbol_map[target]:
                new_lines.append(indent + symbol_map[target])
                continue
            if target in COMMON_STD_LIBS:
                new_lines.append(indent + COMMON_STD_LIBS[target])
                continue

        new_lines.append(line)

    return "\n".join(new_lines)


def get_actually_used_symbols(code: str) -> Set[str]:
    """
    Use AST to detect symbols that are actually used (not just defined).
    More accurate than regex-based detection.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to regex-based detection if AST parsing fails
        return detect_used_symbols(code)

    used_symbols: Set[str] = set()
    defined_symbols: Set[str] = set()

    for node in ast.walk(tree):
        # Track defined names (functions, classes, variables)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defined_symbols.add(node.name)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined_symbols.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For expressions like `obj.attr`, track `obj`
            if isinstance(node.value, ast.Name):
                used_symbols.add(node.value.id)

    # Return symbols that are used but not defined locally
    return used_symbols - defined_symbols


def inject_missing_imports(code: str, symbol_map: Dict[str, str]) -> str:
    """
    Inject missing imports for symbols that are actually used.
    Uses AST-based detection for accuracy.
    """
    symbol_map = symbol_map or {}
    used_symbols = get_actually_used_symbols(code)
    missing_imports: List[str] = []

    def is_imported(symbol: str, source: str) -> bool:
        pattern = rf"^\s*(?!#)(?:import\s+[^\n]*\b{re.escape(symbol)}\b|from\s+[^\n]+\s+import\s+[^\n]*\b{re.escape(symbol)}\b)"
        return re.search(pattern, source, re.MULTILINE) is not None

    for sym in used_symbols:
        # Skip Python keywords and builtins
        if sym in PYTHON_KEYWORDS:
            continue
        if is_imported(sym, code):
            continue
        imp_stmt = None
        if sym in symbol_map:
            imp_stmt = symbol_map[sym]
        elif sym in COMMON_STD_LIBS:
            imp_stmt = COMMON_STD_LIBS[sym]
        if imp_stmt and imp_stmt not in missing_imports and imp_stmt not in code:
            missing_imports.append(imp_stmt)
            print(f"    [Auto-Fix] Injecting import: {imp_stmt}")

    if not missing_imports: return code

    lines = code.splitlines()
    future_imports = []
    other_lines = []
    for line in lines:
        if line.strip().startswith("from __future__"):
            future_imports.append(line)
        else:
            other_lines.append(line)

    return '\n'.join(future_imports + missing_imports + other_lines)


def ensure_root_import(code: str, module_name: str) -> str:
    if not module_name: return code
    root_pkg = module_name.split('.')[0]
    if root_pkg and f"{root_pkg}." in code:
        if not re.search(rf"^\s*(?!#)(import|from)\s+{re.escape(root_pkg)}\b", code, re.MULTILINE):
            lines = code.splitlines()
            insert_idx = 0
            for i, l in enumerate(lines):
                if l.startswith("from __future__"): insert_idx = i + 1
            lines.insert(insert_idx, f"import {root_pkg}")
            return "\n".join(lines)
    return code


def validate_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python syntax using AST parsing.
    Returns (is_valid, error_message).
    """
    if not code or not code.strip():
        return False, "Empty code"

    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_msg = f"Line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f" | {e.text.strip()}"
        return False, error_msg
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def remove_unused_imports(code: str) -> str:
    """
    Remove imports that are not actually used in the code.
    Uses AST-based analysis for accuracy.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code has syntax errors, don't try to clean imports
        return code

    # Collect all imported names
    imported_names: Dict[str, int] = {}  # name -> line_number
    import_lines: Set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_lines.add(node.lineno)
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split('.')[0]
                imported_names[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            import_lines.add(node.lineno)
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = node.lineno

    # Collect all used names (excluding imports)
    used_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Only count if not in an import statement
            if hasattr(node, 'lineno') and node.lineno not in import_lines:
                used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For `module.func()`, track `module`
            if isinstance(node.value, ast.Name):
                if hasattr(node.value, 'lineno') and node.value.lineno not in import_lines:
                    used_names.add(node.value.id)

    # Find unused imports
    unused_names = set(imported_names.keys()) - used_names

    if not unused_names:
        return code

    # Remove unused import lines
    lines = code.splitlines()
    cleaned_lines = []

    for i, line in enumerate(lines, start=1):
        line_stripped = line.strip()

        # Check if this line imports only unused symbols
        should_remove = False
        if line_stripped.startswith(('import ', 'from ')):
            # Extract imported names from this line
            line_imports = set()
            if line_stripped.startswith('import '):
                match = re.search(r'import\s+([\w\.]+)(?:\s+as\s+([\w]+))?', line_stripped)
                if match:
                    name = match.group(2) if match.group(2) else match.group(1).split('.')[0]
                    line_imports.add(name)
            elif line_stripped.startswith('from '):
                match = re.search(r'from\s+[\w\.]+\s+import\s+(.+)', line_stripped)
                if match:
                    imports_str = match.group(1)
                    for imp in imports_str.split(','):
                        imp = imp.strip()
                        if ' as ' in imp:
                            name = imp.split(' as ')[1].strip()
                        else:
                            name = imp.split()[0]
                        line_imports.add(name)

            # Only remove if ALL imports on this line are unused
            if line_imports and line_imports.issubset(unused_names):
                should_remove = True
                print(f"    [Auto-Fix] Removing unused import: {line_stripped}")

        if not should_remove:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def self_correct_probe(probe_code: str, valid_imports_block: str, module_name: str) -> str:
    """
    Main entry point for probe code repair.
    Enhanced with syntax validation and unused import cleanup.
    """
    if not probe_code: return ""

    # 1. Clean & Normalize (Fix Indentation & Markdown)
    cleaned = clean_llm_code(probe_code)

    # 2. Fix Symbol Typos (missing_ -> missing, but protect class_)
    cleaned = fix_symbol_typos(cleaned)

    symbol_map = parse_valid_symbols(valid_imports_block)

    # 3. Fix Bad Imports (Conservative Path Correction)
    cleaned = validate_and_fix_imports(cleaned, module_name, symbol_map)

    # 4. Inject Missing Imports (AST-based detection)
    cleaned = inject_missing_imports(cleaned, symbol_map)

    # 5. Ensure root package import if needed
    cleaned = ensure_root_import(cleaned, module_name)

    # 6. Remove unused imports (NEW)
    cleaned = remove_unused_imports(cleaned)

    # 7. Validate syntax (NEW)
    is_valid, error_msg = validate_syntax(cleaned)
    if not is_valid:
        print(f"    [WARN] Generated code has syntax error: {error_msg}")
        # Return the code anyway - the test runner will catch it
        # This allows us to see what went wrong

    return cleaned
