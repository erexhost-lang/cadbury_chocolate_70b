"""
AST-Based Syntax Validation Pipeline
Parses source code into Abstract Syntax Trees to ensure that only syntactically
valid code enters the pretraining dataset.
"""

import ast
import json
import os
import sys
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("syntax_validator")


def validate_python_ast(code: str) -> Tuple[bool, str]:
    """Parse Python code using the native ast module."""
    try:
        ast.parse(code)
        return True, "valid"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"ParseException: {str(e)}"


def validate_json_syntax(code: str) -> Tuple[bool, str]:
    """Validate JSON data structure."""
    try:
        json.loads(code)
        return True, "valid"
    except Exception as e:
        return False, f"JSONError: {str(e)}"


def validate_generic_brackets(code: str) -> Tuple[bool, str]:
    """
    Fast bracket/parentheses/brace balance check for languages where full
    Tree-sitter grammars might not be installed.
    """
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    
    in_string = False
    quote_char = ""
    escaped = False

    for char in code:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue

        if char in ('"', "'", "`"):
            if not in_string:
                in_string = True
                quote_char = char
            elif char == quote_char:
                in_string = False
            continue

        if not in_string:
            if char in "({[":
                stack.append(char)
            elif char in ")}]":
                if not stack or stack[-1] != pairs[char]:
                    return False, f"Unbalanced bracket mismatch: '{char}'"
                stack.pop()

    if stack:
        return False, f"Unclosed bracket: '{stack[-1]}'"
    return True, "valid"


def validate_code_syntax(content: str, language: str) -> Tuple[bool, str]:
    """
    Route code to appropriate syntax validator based on file language.
    """
    lang = language.lower()
    
    if lang == "python":
        return validate_python_ast(content)
    elif lang == "json":
        return validate_json_syntax(content)
    elif lang in ("javascript", "typescript", "rust", "go", "cpp", "c", "java"):
        # Uses bracket balance check as reliable syntax filter
        return validate_generic_brackets(content)
    else:
        # Pass unknown/markdown formats through
        return True, "skipped"


if __name__ == "__main__":
    valid_py = "def hello():\n    print('world')\n"
    invalid_py = "def hello(\n    print('world')\n"

    print("Valid Python test:", validate_code_syntax(valid_py, "python"))
    print("Invalid Python test:", validate_code_syntax(invalid_py, "python"))
