"""
GitHub Repository Code Harvester
Crawls repositories, ignores binary/generated/vendored files,
and yields clean source code documents with metadata.
"""

import argparse
import json
import os
import sys
from typing import Generator, Dict, Any, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("github_crawler")

# File extensions mapped to language classification
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".sql": "sql",
    ".sh": "bash",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
}

# Directories to strictly exclude (vendored, compiled, cache, minified)
EXCLUDE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
    ".cache",
}

# Binary and non-code file extensions to ignore
EXCLUDE_EXTS: Set[str] = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".class",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".7z", ".mp4", ".wav", ".lock",
    ".min.js", ".min.css", ".map"
}


def is_binary_string(bytes_chunk: bytes) -> bool:
    """Check for null bytes, typical in compiled or binary data."""
    return b"\x00" in bytes_chunk


def harvest_local_or_remote_repos(root_dir: str) -> Generator[Dict[str, Any], None, None]:
    """
    Recursively walk through directory tree and yield valid source code files.
    """
    if not os.path.exists(root_dir):
        logger.error(f"Root path '{root_dir}' does not exist.")
        return

    logger.info(f"Scanning directory: {root_dir}")
    file_count = 0

    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories in-place to avoid wasting I/O
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

        for fname in files:
            _, ext = os.path.splitext(fname)
            ext = ext.lower()

            if ext in EXCLUDE_EXTS or ext not in EXT_TO_LANG:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, root_dir)

            try:
                # Read sample to verify non-binary encoding
                with open(full_path, "rb") as f:
                    sample = f.read(2048)
                    if is_binary_string(sample):
                        continue

                # Read full text
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Basic quality filter: ignore empty or trivially small files
                if len(content.strip()) < 50:
                    continue

                file_count += 1
                yield {
                    "source": "github_crawler",
                    "language": EXT_TO_LANG[ext],
                    "filepath": rel_path,
                    "content": content,
                    "size_bytes": len(content.encode("utf-8")),
                }

            except Exception as e:
                logger.debug(f"Failed reading {full_path}: {e}")
                continue

    logger.info(f"Completed scan of '{root_dir}'. Harvested {file_count:,} source code files.")


def main():
    parser = argparse.ArgumentParser(description="Harvest code files from git repositories")
    parser.add_argument("--repo-dir", type=str, required=True, help="Path to local repository root")
    parser.add_argument("--output", type=str, default="data/raw/harvested_repos.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    count = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in harvest_local_or_remote_repos(args.repo_dir):
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Successfully harvested {count} files -> {args.output}")


if __name__ == "__main__":
    main()
