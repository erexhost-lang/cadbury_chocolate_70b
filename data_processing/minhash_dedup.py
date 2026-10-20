"""
MinHash & Locality-Sensitive Hashing (LSH) Near-Deduplication
Filters out near-identical duplicate files (forks, copied boilerplates,
common license headers) to prevent memorization and bias.
"""

import re
import sys
import os
from typing import List, Set, Generator, Dict, Any, Tuple
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("minhash_dedup")


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """
    Generate character or token k-shingles from normalized text.
    Removes whitespace variations to focus on structural syntax.
    """
    # Normalize whitespaces to single spaces
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    words = normalized.split(" ")
    
    if len(words) < k:
        return set(words)
    
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class MinHashDeduplicator:
    """Manages LSH index and identifies near-duplicate code files."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        """
        Args:
            threshold: Jaccard similarity cutoff (0.85 = 85% duplicate content)
            num_perm: Number of hash permutations for MinHash signature
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_keys = set()

    def compute_minhash(self, text: str) -> MinHash:
        """Create MinHash fingerprint from text shingles."""
        m = MinHash(num_perm=self.num_perm)
        for shingle in get_shingles(text):
            m.update(shingle.encode("utf-8"))
        return m

    def is_duplicate(self, doc_id: str, text: str) -> bool:
        """
        Check if document is near-duplicate of an existing document in the index.
        If unique, inserts into index and returns False.
        """
        if len(text.strip()) < 50:
            return True  # Filter trivial files

        minhash = self.compute_minhash(text)
        candidates = self.lsh.query(minhash)

        if candidates:
            return True

        # Insert new unique document into index
        try:
            self.lsh.insert(doc_id, minhash)
            self.seen_keys.add(doc_id)
        except ValueError:
            # Key already in index
            return True

        return False


def deduplicate_dataset_stream(
    records: Generator[Dict[str, Any], None, None],
    threshold: float = 0.85,
) -> Generator[Dict[str, Any], None, None]:
    """Generator filter that yields only unique, non-duplicate records."""
    deduper = MinHashDeduplicator(threshold=threshold)
    total = 0
    duplicates = 0

    for idx, rec in enumerate(records):
        total += 1
        doc_id = rec.get("filepath", f"doc_{idx}")
        content = rec.get("content", "")

        if deduper.is_duplicate(doc_id, content):
            duplicates += 1
            continue

        yield rec

        if total % 5000 == 0:
            logger.info(
                f"Processed {total:,} records | Duplicates removed: {duplicates:,} "
                f"({(duplicates/total)*100:.1f}%)"
            )


if __name__ == "__main__":
    code_v1 = """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    """

    # Slightly modified copy (variable rename)
    code_v2 = """
    def calculate_fibonacci(num):
        if num <= 1:
            return num
        return calculate_fibonacci(num-1) + calculate_fibonacci(num-2)
    """

    deduper = MinHashDeduplicator(threshold=0.85)
    print("Is v1 duplicate?", deduper.is_duplicate("file_1", code_v1))
    print("Is v2 duplicate of v1?", deduper.is_duplicate("file_2", code_v2))
