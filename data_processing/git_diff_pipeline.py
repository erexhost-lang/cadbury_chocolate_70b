"""
Git Diff, Commit, and Pull Request Evolution Pipeline
Transforms software evolution trajectories (Git commits, issues, and PRs) into training samples:
  [Issue / Commit Intent] + [Target File Slice] -> [<think> Ripple Analysis] -> [Search/Replace Patch]

Teaches the model surgical SEARCH/REPLACE editing syntax:
<<<<<<< SEARCH
exact lines to replace
=======
new updated code lines
>>>>>>> REPLACE
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("git_diff_pipeline")

SEARCH_BLOCK_START = "<<<<<<< SEARCH"
SEARCH_DIVIDER = "======="
REPLACE_BLOCK_END = ">>>>>>> REPLACE"


@dataclass
class SearchReplaceHunk:
    file_path: str
    search_content: str
    replace_content: str


def parse_search_replace_blocks(patch_text: str) -> List[SearchReplaceHunk]:
    """
    Parses SEARCH/REPLACE blocks formatted like:
    # File: path/to/file.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE
    """
    hunks = []
    file_regex = re.compile(r"^\s*#\s*File:\s*([^\s\n]+)", re.MULTILINE)
    
    # Split by SEARCH/REPLACE patterns
    pattern = re.compile(
        r"(?:#\s*File:\s*([^\s\n]+)\s*\n)?"
        r"<<<<<<< SEARCH\n([\s\S]*?)\n"
        r"=======\n([\s\S]*?)\n"
        r">>>>>>> REPLACE",
        re.MULTILINE,
    )

    current_file = "unknown"
    for match in pattern.finditer(patch_text):
        file_path_match, search_text, replace_text = match.groups()
        if file_path_match:
            current_file = file_path_match
        hunks.append(
            SearchReplaceHunk(
                file_path=current_file,
                search_content=search_text,
                replace_content=replace_text,
            )
        )

    return hunks


def apply_search_replace_patch(original_code: str, hunks: List[SearchReplaceHunk]) -> Tuple[bool, str, str]:
    """
    Applies surgical search/replace hunks to source code.
    Verifies that search blocks match uniquely in the original code.
    Returns:
        (success, updated_code, error_message)
    """
    modified = original_code
    for idx, hunk in enumerate(hunks):
        search_str = hunk.search_content
        replace_str = hunk.replace_content

        count = modified.count(search_str)
        if count == 0:
            return False, original_code, f"Hunk {idx+1}: SEARCH block not found in target file."
        if count > 1:
            return False, original_code, f"Hunk {idx+1}: Ambiguous match ({count} occurrences) in target file."

        modified = modified.replace(search_str, replace_str, 1)

    return True, modified, ""


def format_commit_pr_training_sample(
    repo_name: str,
    commit_msg_or_issue: str,
    original_files: Dict[str, str],
    modified_files: Dict[str, str],
    reasoning_analysis: str,
) -> Dict[str, Any]:
    """
    Generates a high-signal surgical training record pairing:
    - User request (Commit intent / Issue description)
    - Source code context (Target file slice)
    - Thought process (Impact & ripple analysis)
    - Output (Minimal SEARCH/REPLACE diff blocks)
    """
    # 1. Generate surgical diff blocks
    diff_blocks = []
    for file_path, new_code in modified_files.items():
        old_code = original_files.get(file_path, "")
        diff_blocks.append(f"# File: {file_path}")

        # Extract differing lines (simple extraction for demonstration)
        diff_blocks.append(f"{SEARCH_BLOCK_START}\n{old_code.strip()}\n{SEARCH_DIVIDER}\n{new_code.strip()}\n{REPLACE_BLOCK_END}")

    patch_output = "\n\n".join(diff_blocks)

    system_prompt = (
        "You are an expert Software Engineer who writes minimal, surgical code patches. "
        "Analyze the architectural ripple effects inside <think>...</think> tags, then format "
        "your patch strictly using <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
    )

    user_prompt = (
        f"Repository: {repo_name}\n"
        f"Task Description:\n{commit_msg_or_issue}\n\n"
        "Target Files for Modification:\n" + "\n".join(f"- {p}" for p in original_files.keys())
    )

    assistant_response = f"<think>\n{reasoning_analysis}\n</think>\n\n```patch\n{patch_output}\n```"

    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": {
            "repo": repo_name,
            "modified_files_count": len(modified_files),
            "format": "search_replace_diff",
        },
    }


if __name__ == "__main__":
    print("--- Testing Search/Replace Diff Pipeline Locally ---")

    orig_py = """def calculate_tax(income):
    # Old tax calculation
    tax_rate = 0.20
    return income * tax_rate
"""

    patch = """# File: finance/tax.py
<<<<<<< SEARCH
    # Old tax calculation
    tax_rate = 0.20
    return income * tax_rate
=======
    # New progressive tax calculation (2026 update)
    if income > 100000:
        return income * 0.30
    return income * 0.15
>>>>>>> REPLACE"""

    hunks = parse_search_replace_blocks(patch)
    print(f"Parsed {len(hunks)} search/replace hunk(s) for file: {hunks[0].file_path}")

    ok, updated, err = apply_search_replace_patch(orig_py, hunks)
    print("Patch application success:", ok)
    print("Updated Code Result:\n" + updated)
