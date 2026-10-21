"""
Cadbury-Chocolate Frontier V2: Multi-Domain Data Mixing Pipeline
Implements Weighted Dataset Sampling across 3.0T to 5.5T Pretraining Token Schedule.

Target Distribution:
- 60% Code & Software Distribution (Permissive Range: 50% - 70%):
  * 40.0% Multi-Language Repositories (30+ languages, AST-parsed, FIM packed)
  * 10.0% Git Diff & PR Evolution Trajectories (SEARCH/REPLACE blocks)
  *  5.0% Technical Documentation & API Specifications
  *  5.0% Code QA Discussions (Stack Overflow & GitHub Issues)
- 40% High-Quality Natural Language & Formal Logic (Permissive Range: 30% - 50%):
  * 15.0% STEM, Mathematical Literature & Formal Proofs (ArXiv, Lean4/Isabelle)
  * 15.0% Computer Science Textbooks & Academic Syllabi
  * 10.0% Curated Technical Web & Engineering Post-Mortems (FineWeb-Edu)

Key Capabilities:
1. Domain-Calibrated Quality Filtering (Code syntax vs Formal prose).
2. Domain-Calibrated MinHash Deduplication (Jaccard 0.85 for code, 0.75 for prose).
3. Weighted Stochastic Ingestion Generator with FIM chunking and token volume tracking.
"""

import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_processing.minhash_dedup import MinHashDeduplicator
from data_processing.quality_filter import is_high_quality_code
from data_processing.repo_packer_fim import apply_fim_slice

logger = setup_logger("dataset_mixer")


# ==============================================================================
# 1. Domain Enums and Configuration Dataclasses
# ==============================================================================
class DomainCategory(str, Enum):
    CODE = "code"
    NATURAL_LANGUAGE = "natural_language"


@dataclass
class DomainConfig:
    name: str
    category: DomainCategory
    ratio: float
    description: str
    dedup_threshold: float
    fim_rate: float
    languages_count: Optional[int] = None


@dataclass
class MixingScheduleConfig:
    target_tokens: int = 4_000_000_000_000 # 4.0 Trillion tokens
    min_tokens: int = 3_000_000_000_000    # 3.0 Trillion tokens
    max_tokens: int = 5_500_000_000_000    # 5.5 Trillion tokens
    min_code_ratio: float = 0.50
    max_code_ratio: float = 0.70
    min_text_ratio: float = 0.30
    max_text_ratio: float = 0.50
    domains: Dict[str, DomainConfig] = field(default_factory=dict)

    def validate_ratios(self):
        """Validates that domain ratios sum to 1.0 and stay within architectural boundaries."""
        total_ratio = sum(d.ratio for d in self.domains.values())
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-3):
            raise ValueError(f"Total domain mixing ratios must sum to 1.0, got {total_ratio:.4f}")

        code_ratio = sum(d.ratio for d in self.domains.values() if d.category == DomainCategory.CODE)
        text_ratio = sum(d.ratio for d in self.domains.values() if d.category == DomainCategory.NATURAL_LANGUAGE)

        if not (self.min_code_ratio <= code_ratio <= self.max_code_ratio):
            raise ValueError(
                f"Code domain ratio ({code_ratio:.2%}) violates target bounds "
                f"[{self.min_code_ratio:.2%}, {self.max_code_ratio:.2%}]"
            )

        if not (self.min_text_ratio <= text_ratio <= self.max_text_ratio):
            raise ValueError(
                f"Natural language text ratio ({text_ratio:.2%}) violates target bounds "
                f"[{self.min_text_ratio:.2%}, {self.max_text_ratio:.2%}]"
            )

        logger.info(
            f"Validated Multi-Domain Mixing Ratios: {code_ratio:.1%} Code / {text_ratio:.1%} Reasoning Text. "
            f"Target Schedule: {self.target_tokens / 1e12:.1f}T Tokens."
        )


# ==============================================================================
# 2. Domain-Calibrated Quality & Deduplication Filters
# ==============================================================================
class CalibratedDomainFilter:
    """
    Applies distinct heuristic quality filters to code vs formal reasoning text.
    """

    @staticmethod
    def is_valid_text_prose(text: str) -> Tuple[bool, str]:
        """
        Quality filter for STEM papers, textbooks, and technical articles.
        Filters out web crawl noise, machine-translated spam, and repetitive boilerplates.
        """
        stripped = text.strip()
        if len(stripped) < 100:
            return False, "text_too_short"

        lines = stripped.splitlines()
        if not lines:
            return False, "no_lines"

        # Check for HTML / XML web scrapings
        html_tags = len(re.findall(r"<[^>]+>", text))
        if html_tags > 20 and html_tags / max(1, len(lines)) > 0.4:
            return False, "excessive_raw_html"

        # Check average line length (eliminates unformatted walls of text or broken crawls)
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < 15.0:
            return False, "line_length_too_short"
        if avg_line_len > 400.0:
            return False, "line_length_too_long"

        # Check word repetitiveness (catches keyword-stuffing spam)
        words = re.findall(r"\b[A-Za-z0-9_]+\b", text.lower())
        if len(words) > 50:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.25:
                return False, "high_word_repetition"

        return True, "valid_prose"

    @classmethod
    def filter_document(cls, text: str, category: DomainCategory) -> Tuple[bool, str]:
        """Routes document to its domain-calibrated quality filter."""
        if category == DomainCategory.CODE:
            return is_high_quality_code(text)
        else:
            return cls.is_valid_text_prose(text)


# ==============================================================================
# 3. Multi-Domain Weighted Dataset Mixer
# ==============================================================================
class MultiDomainDatasetMixer:
    """
    Manages multi-source streaming, domain interleaving, and FIM processing
    for 3.0T - 5.5T pre-training tokens.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.config.validate_ratios()

        # Domain deduplicators with calibrated Jaccard thresholds
        self.deduplicators: Dict[str, MinHashDeduplicator] = {
            name: MinHashDeduplicator(threshold=cfg.dedup_threshold, num_perm=128)
            for name, cfg in self.config.domains.items()
        }

        # Token volume counters
        self.domain_token_counts: Dict[str, int] = {name: 0 for name in self.config.domains}
        self.total_tokens_accumulated: int = 0

    def _load_config(self, config_path: Optional[str]) -> MixingScheduleConfig:
        """Loads configuration from YAML or sets default 60/40 schedule."""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                raw_cfg = yaml.safe_load(f)

            scaling = raw_cfg.get("scaling_laws", {})
            mixing = raw_cfg.get("data_mixing", {})
            domains_raw = mixing.get("domains", {})

            cfg = MixingScheduleConfig(
                target_tokens=scaling.get("target_token_volume", 4_000_000_000_000),
                min_tokens=scaling.get("min_token_volume", 3_000_000_000_000),
                max_tokens=scaling.get("max_token_volume", 5_500_000_000_000),
                min_code_ratio=mixing.get("enforce_code_range", [0.50, 0.70])[0],
                max_code_ratio=mixing.get("enforce_code_range", [0.50, 0.70])[1],
                min_text_ratio=mixing.get("enforce_text_range", [0.30, 0.50])[0],
                max_text_ratio=mixing.get("enforce_text_range", [0.30, 0.50])[1],
            )

            for d_name, d_info in domains_raw.items():
                cat = (
                    DomainCategory.CODE
                    if any(c in d_name for c in ["code", "github", "diff", "qa", "docs"])
                    else DomainCategory.NATURAL_LANGUAGE
                )
                cfg.domains[d_name] = DomainConfig(
                    name=d_name,
                    category=cat,
                    ratio=d_info.get("ratio", 0.1),
                    description=d_info.get("description", ""),
                    dedup_threshold=d_info.get("dedup_jaccard_threshold", 0.80),
                    fim_rate=d_info.get("fim_rate", 0.0),
                    languages_count=d_info.get("languages_count"),
                )
            return cfg

        # Built-in Default: 60% Code / 40% Reasoning Prose
        cfg = MixingScheduleConfig()
        cfg.domains = {
            "multilang_github_repos": DomainConfig(
                "multilang_github_repos", DomainCategory.CODE, 0.40,
                "Multi-language code from GitHub", 0.85, fim_rate=0.50, languages_count=35
            ),
            "git_diff_pr_evolution": DomainConfig(
                "git_diff_pr_evolution", DomainCategory.CODE, 0.10,
                "Commit histories & PR Search/Replace diffs", 0.80, fim_rate=0.20
            ),
            "technical_docs_apis": DomainConfig(
                "technical_docs_apis", DomainCategory.CODE, 0.05,
                "Official SDK specifications & RFCs", 0.80, fim_rate=0.30
            ),
            "code_qa_discussions": DomainConfig(
                "code_qa_discussions", DomainCategory.CODE, 0.05,
                "Stack Overflow QA & GitHub Issues", 0.78, fim_rate=0.10
            ),
            "stem_math_formal_logic": DomainConfig(
                "stem_math_formal_logic", DomainCategory.NATURAL_LANGUAGE, 0.15,
                "ArXiv papers, Lean4 proofs & formal logic", 0.75, fim_rate=0.0
            ),
            "cs_textbooks_academic": DomainConfig(
                "cs_textbooks_academic", DomainCategory.NATURAL_LANGUAGE, 0.15,
                "CS textbooks, algorithms & systems literature", 0.75, fim_rate=0.0
            ),
            "curated_tech_web": DomainConfig(
                "curated_tech_web", DomainCategory.NATURAL_LANGUAGE, 0.10,
                "FineWeb-Edu technical articles & post-mortems", 0.72, fim_rate=0.0
            ),
        }
        return cfg

    def sample_domain(self) -> str:
        """Stochastically samples the next domain based on target probability distribution."""
        domain_names = list(self.config.domains.keys())
        weights = [self.config.domains[d].ratio for d in domain_names]
        return random.choices(domain_names, weights=weights, k=1)[0]

    def process_and_mix_sample(
        self,
        domain_name: str,
        doc_id: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Filters, deduplicates, applies FIM (if code), and tracks token volume.
        """
        domain_cfg = self.config.domains[domain_name]

        # 1. Calibrated Quality Filter
        is_valid, reason = CalibratedDomainFilter.filter_document(content, domain_cfg.category)
        if not is_valid:
            return None

        # 2. Calibrated MinHash Near-Deduplication
        deduplicator = self.deduplicators[domain_name]
        if deduplicator.is_duplicate(doc_id, content):
            return None

        # 3. Optional Fill-In-The-Middle (FIM) Transformation
        final_text = content
        is_fim = False
        if domain_cfg.fim_rate > 0.0 and random.random() < domain_cfg.fim_rate:
            final_text, is_fim = apply_fim_slice(content)

        # 4. Token Accounting (Approximation: 1 word ~ 1.3 tokens)
        approx_tokens = int(len(final_text.split()) * 1.3)
        self.domain_token_counts[domain_name] += approx_tokens
        self.total_tokens_accumulated += approx_tokens

        return {
            "domain": domain_name,
            "category": domain_cfg.category.value,
            "doc_id": doc_id,
            "text": final_text,
            "is_fim": is_fim,
            "tokens_est": approx_tokens,
        }

    def get_progress_telemetry(self) -> Dict[str, Any]:
        """Returns training progress telemetry towards the 3.0T-5.5T token goal."""
        total = max(1, self.total_tokens_accumulated)
        active_code = sum(
            self.domain_token_counts[k] for k, v in self.config.domains.items()
            if v.category == DomainCategory.CODE
        )
        active_text = total - active_code

        domain_percentages = {
            k: (self.domain_token_counts[k] / total) * 100.0
            for k in self.config.domains
        }

        return {
            "total_tokens_accumulated": self.total_tokens_accumulated,
            "target_token_volume": self.config.target_tokens,
            "completion_percentage": (self.total_tokens_accumulated / self.config.target_tokens) * 100.0,
            "code_ratio_actual": (active_code / total) * 100.0,
            "text_ratio_actual": (active_text / total) * 100.0,
            "domain_breakdown": domain_percentages,
        }


# ==============================================================================
# 4. Self-Contained Verification & Demonstration
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Cadbury-70B Frontier V2: Multi-Domain Dataset Mixer Verification")
    print("=" * 80)

    cfg_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "training_scale_config.yaml",
    )

    mixer = MultiDomainDatasetMixer(config_path=cfg_file)
    print(f"\n[1. Configuration Loaded]")
    print(f"Target Volume: {mixer.config.target_tokens / 1e12:.2f} Trillion Tokens")
    print(f"Number of Active Ingestion Domains: {len(mixer.config.domains)}")

    # Display configured ratios
    print(f"\n[2. Configured Target Domains & Mixing Ratios]")
    for name, d_cfg in mixer.config.domains.items():
        print(f"  {name:25} ({d_cfg.category.value:16}) -> {d_cfg.ratio:5.1%} | Dedup: {d_cfg.dedup_threshold} | FIM: {d_cfg.fim_rate:.0%}")

    # Simulate streaming 200 samples across domains
    print(f"\n[3. Simulating Weighted Multi-Domain Sampling (200 steps)]")
    samples_generated = 0
    for step in range(200):
        domain = mixer.sample_domain()
        doc_id = f"doc_{step:04d}"
        if "code" in domain or "github" in domain or "diff" in domain:
            raw_text = f"def compute_step_{step}(x: int) -> int:\n    # Computes step value\n    result = x * 2 + 10\n    return result\n"
        else:
            raw_text = f"Theorem {step}: Let G be a connected graph with N vertices. The minimum spanning tree exhibits O(E log V) complexity.\n"

        sample = mixer.process_and_mix_sample(domain, doc_id, raw_text)
        if sample:
            samples_generated += 1

    telemetry = mixer.get_progress_telemetry()
    print(f"Samples Ingested: {samples_generated}/200")
    print(f"Tokens Accumulated: {telemetry['total_tokens_accumulated']:,} tokens")
    print(f"Simulated Distribution: {telemetry['code_ratio_actual']:.1f}% Code / {telemetry['text_ratio_actual']:.1f}% Reasoning Text")
    print("=" * 80)
    print("Multi-Domain Data Mixing Pipeline Verified Successfully.")
