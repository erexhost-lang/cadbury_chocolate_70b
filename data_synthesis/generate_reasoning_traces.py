"""
Synthetic Chain-of-Thought (CoT) and Agentic Data Pipeline
Enhanced with:
1. Bayesian Hypothesis Localization: Formulates ranked probabilities before file inspection.
2. Surgical Context Rules: Forbids blind dumps, mandates targeted line-range slices.
3. Anti-Rumination Filter: Detects and eliminates circular backtracking and self-doubt loops.
"""

import json
import os
import re
import sys
from typing import List, Dict, Any, Generator, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_synthesis.schemas import (
    THINK_START,
    THINK_END,
    HYPOTHESIS_START,
    HYPOTHESIS_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_OUTPUT_START,
    TOOL_OUTPUT_END,
    RoleEnum,
    TrackEnum,
    ChatMessage,
    ToolCall,
    ToolOutput,
    ReasoningTrajectory,
    SystemDesignSpec,
    DSASpec,
    DebuggingAgentSpec,
    RankedHypothesisItem,
    BayesianHypothesis,
)

logger = setup_logger("generate_reasoning_traces")

# Patterns associated with low-quality rumination and circular self-doubt
RUMINATION_PATTERNS = [
    re.compile(r"(?:wait,\s+let\s+me\s+re-?verify|let\s+me\s+double\s+check\s+again|wait,\s+actually\s+maybe)", re.IGNORECASE),
    re.compile(r"(?:actually,\s+on\s+second\s+thought,\s+maybe\s+not)", re.IGNORECASE),
    re.compile(r"(?:let\s+me\s+re-?read\s+the\s+entire\s+prompt\s+carefully)", re.IGNORECASE),
    re.compile(r"(?:wait,\s+is\s+that\s+right\?\s+let\s+me\s+rethink)", re.IGNORECASE),
]


def anti_rumination_filter(thought: str, max_backtracks: int = 1) -> Tuple[bool, str]:
    """
    Scans a reasoning trace for unproductive circular self-doubt loops.
    Returns:
        (is_clean, sanitized_or_flagged_thought)
    """
    total_rumination_hits = 0
    cleaned = thought

    for pattern in RUMINATION_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            total_rumination_hits += len(matches)
            # Remove repetitive filler phrases to streamline context signal-to-noise ratio
            cleaned = pattern.sub("", cleaned)

    # Check for excessive repetition of the word "wait" or "actually"
    wait_count = len(re.findall(r"\bwait\b", cleaned, re.IGNORECASE))
    if wait_count > 3:
        return False, "Excessive circular self-doubt loops detected."

    if total_rumination_hits > max_backtracks:
        return False, f"Rumination threshold exceeded ({total_rumination_hits} circular backtrack phrases)."

    # Normalize multiple whitespace created by pruning
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return True, cleaned


class SyntheticReasoningPipeline:
    """Generates and serializes structured reasoning trajectories for SFT."""

    def __init__(self, output_dir: str = "data/processed/sft_reasoning"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ==========================================================================
    # Track 1: System Design & Fault Tolerance Track
    # ==========================================================================
    def build_system_design_sample(self, spec: SystemDesignSpec) -> ReasoningTrajectory:
        system_prompt = (
            "You are a Principal Distributed Systems Engineer. When designing backend "
            "infrastructure, you must first reason deeply inside <think>...</think> tags "
            "evaluating single points of failure, network partitions, cache invalidation, "
            "and load shedding before writing infrastructure or backend code."
        )

        user_content = (
            f"Design a resilient, fault-tolerant backend service: '{spec.service_name}'.\n"
            f"- Target Throughput: {spec.target_rps:,} RPS\n"
            f"- Target Latency SLA: P99 < {spec.p99_latency_ms} ms\n"
            f"- Storage: {spec.storage_architecture}\n"
            f"- Compute: {spec.compute_infrastructure}\n"
            f"- Requirements: {', '.join(spec.fault_tolerance_requirements)}\n"
            "Provide the complete architectural analysis, failure modes, and Kubernetes/Go/Terraform implementation."
        )

        thought_process = f"""1. Capacity & Bottleneck Sizing:
- Target is {spec.target_rps:,} RPS. Ingress bandwidth is ~{spec.target_rps * 1 / 1000:.1f} MB/s (~{spec.target_rps * 8 / 1000:.1f} Gbps).
- PostgreSQL direct writes saturate at ~10k RPS. Solution: Buffer with partitioned Kafka/Redpanda topic.

2. Single Point of Failure (SPOF) Analysis:
- Cache Stampede: Use probabilistic early expiration (XFetch algorithm) + Redis multi-AZ read replicas.
- Raft consensus for stateful transactions; AP model for telemetry streams.

3. Resilience & Graceful Degradation:
- Envoy proxy adaptive concurrency limits + token bucket rate limiting.
- Circuit breaker: Trip when 5xx error rate exceeds 5% over 10s sliding window."""

        # Verify anti-rumination rules
        is_clean, sanitized_thought = anti_rumination_filter(thought_process)
        if not is_clean:
            thought_process = sanitized_thought

        assistant_code_response = f"""### Architectural Implementation: {spec.service_name}

```go
package main

import (
	"context"
	"time"
	"github.com/sony/gobreaker"
)

type ResilientService struct {
	cb *gobreaker.CircuitBreaker
}

func NewResilientService() *ResilientService {
	return &ResilientService{
		cb: gobreaker.NewCircuitBreaker(gobreaker.Settings{
			Name:        "{spec.service_name}",
			MaxRequests: 5,
			Interval:    10 * time.Second,
			Timeout:     5 * time.Second,
		}),
	}
}
```"""

        messages = [
            ChatMessage(role=RoleEnum.USER, content=user_content),
            ChatMessage(
                role=RoleEnum.ASSISTANT,
                content=assistant_code_response,
                thought=thought_process,
            ),
        ]

        return ReasoningTrajectory(
            trajectory_id=f"sys_design_{spec.service_name.lower().replace(' ', '_')}",
            track=TrackEnum.SYSTEM_DESIGN,
            system_prompt=system_prompt,
            messages=messages,
            metadata={"target_rps": spec.target_rps},
        )

    # ==========================================================================
    # Track 2: DSA & Algorithmic Optimization Track
    # ==========================================================================
    def build_dsa_sample(self, spec: DSASpec) -> ReasoningTrajectory:
        system_prompt = (
            "You are an expert Competitive Programmer. Before providing any code, you must "
            "perform a strict 4-phase algorithmic analysis inside <think>...</think>:\n"
            "1. Constraint & Edge Case Audit\n"
            "2. Brute-Force TLE/MLE Calculation\n"
            "3. Optimal Data Structure & Invariant Selection\n"
            "4. Asymptotic Time & Space Proof"
        )

        user_content = (
            f"Problem: {spec.problem_title}\n\n"
            f"{spec.problem_statement}\n\n"
            f"Constraints:\n" + "\n".join(f"- {c}" for c in spec.constraints)
        )

        thought_process = f"""Phase 1: Constraints & Edge Cases:
- Constraints: {', '.join(spec.constraints)}. N >= 10^5 mandates O(N) or O(N log N).
- Edge Cases: {', '.join(spec.edge_cases)}.

Phase 2: Brute-Force Failure Proof (TLE Analysis):
- Naive algorithm takes {spec.brute_force_time}. Under N=10^5, 10^10 ops -> TLE.

Phase 3: Optimal Invariant Selection:
- Sliding window invariant with hash table frequency tracking yields {spec.target_time_complexity}.

Phase 4: Big-O Complexity Proof:
- Each element processed at most twice by two pointers: Amortized {spec.target_time_complexity}.
- Auxiliary map stores at most K keys: {spec.target_space_complexity}."""

        is_clean, sanitized_thought = anti_rumination_filter(thought_process)
        if not is_clean:
            thought_process = sanitized_thought

        code_solution = f"""```python
from typing import List
from collections import defaultdict

def solve(arr: List[int], k: int) -> int:
    \"\"\"Solves {spec.problem_title} in {spec.target_time_complexity} time.\"\"\"
    if not arr or k <= 0:
        return 0
    left, max_length = 0, 0
    freq_map = defaultdict(int)

    for right in range(len(arr)):
        freq_map[arr[right]] += 1
        while len(freq_map) > k:
            freq_map[arr[left]] -= 1
            if freq_map[arr[left]] == 0:
                del freq_map[arr[left]]
            left += 1
        max_length = max(max_length, right - left + 1)

    return max_length
```"""

        messages = [
            ChatMessage(role=RoleEnum.USER, content=user_content),
            ChatMessage(
                role=RoleEnum.ASSISTANT,
                content=code_solution,
                thought=thought_process,
            ),
        ]

        return ReasoningTrajectory(
            trajectory_id=f"dsa_{spec.problem_title.lower().replace(' ', '_')}",
            track=TrackEnum.DSA_OPTIMIZATION,
            system_prompt=system_prompt,
            messages=messages,
            metadata={"target_time": spec.target_time_complexity},
        )

    # ==========================================================================
    # Track 3: Bayesian Hypothesis Localization & Surgical Debugging
    # ==========================================================================
    def build_tool_debugging_sample(self, spec: DebuggingAgentSpec) -> ReasoningTrajectory:
        """
        Creates a Bayesian Root-Cause Localization trajectory:
        1. Formulates ranked prior probabilities across architectural layers before touching tools.
        2. Strictly uses surgical reads (line slices, not full file dumps).
        3. Never dumps blind directory trees (`ls -R` is forbidden).
        """
        system_prompt = (
            "You are a Surgical Senior Systems Engineer. When diagnosing cross-stack errors:\n"
            "1. Formulate a ranked Bayesian probability hypothesis (<hypothesis>) of root causes before calling tools.\n"
            "2. Never perform blind repository dumping or read entire files when inspecting code.\n"
            "3. Use targeted line slices (e.g. read_file with start_line and end_line) to maintain high Context SNR.\n"
            "4. Reason deliberately inside <think>...</think> without circular rumination."
        )

        user_content = (
            f"Cross-Stack Incident in '{spec.repo_name}':\n"
            f"{spec.bug_description}\n\n"
            f"Stack Trace:\n```\n{spec.stack_trace}\n```\n"
            "Localize the root cause surgically and provide the minimal patch."
        )

        # 1. Structured Bayesian Hypothesis formulated BEFORE any tool call
        bayes_hypo = BayesianHypothesis(
            hypotheses=[
                RankedHypothesisItem(
                    id="H1",
                    probability=0.65,
                    suspect_layer="DB_SCHEMA_MIGRATION",
                    rationale="Error 'KeyError: is_verified' during user authentication indicates user model was queried but column was not populated in SQL schema.",
                    target_file="models/user.py",
                    target_slice=[35, 55],
                ),
                RankedHypothesisItem(
                    id="H2",
                    probability=0.25,
                    suspect_layer="API_SERIALIZER",
                    rationale="Pydantic serializer in schemas/user.py may have 'is_verified' as mandatory without a default value.",
                    target_file="schemas/user.py",
                    target_slice=[15, 30],
                ),
                RankedHypothesisItem(
                    id="H3",
                    probability=0.10,
                    suspect_layer="FRONTEND_STATE",
                    rationale="React state in UserProfile.tsx might be accessing undefined property.",
                    target_file="frontend/UserProfile.tsx",
                ),
            ],
            falsification_plan="Test H1 first by reading models/user.py lines 35-55. If model contains is_verified, evaluate H2 serializer.",
        )

        turn1_thought = (
            "Given the cross-stack error, H1 has 65% prior probability. "
            "Surgical Rule: Do not read the entire 600-line model file. "
            "Inspect only lines 35 to 55 of models/user.py to check column definitions."
        )
        turn1_tool = ToolCall(name="read_file", arguments={"path": "models/user.py", "start_line": 35, "end_line": 55})

        turn1_output = ToolOutput(
            name="read_file",
            output="""35: class User(Base):
36:     __tablename__ = "users"
37:     id = Column(Integer, primary_key=True)
38:     email = Column(String(255), unique=True, nullable=False)
39:     # Note: is_verified was added to migration 004 but missing from SQLAlchemy model!
40:     hashed_password = Column(String(255), nullable=False)""",
            exit_code=0,
        )

        # Turn 2: Falsification verified H1 immediately on first hop
        turn2_thought = (
            "H1 confirmed on hop 1: Line 39 explicitly documents that `is_verified` was omitted "
            "from the SQLAlchemy model attributes. The serializer expects this field, causing KeyError. "
            "Next: Run pytest using bash tool on auth test suite to confirm the exact failure."
        )
        turn2_tool = ToolCall(name="bash", arguments={"command": "pytest tests/test_user_model.py -k test_user_serialization"})

        turn2_output = ToolOutput(
            name="bash",
            output="FAILED tests/test_user_model.py::test_user_serialization - KeyError: 'is_verified'",
            exit_code=1,
        )

        # Turn 3: Minimal surgical patch
        turn3_thought = (
            "Root cause verified. Solution: Add `is_verified = Column(Boolean, default=False, nullable=False)` "
            "to models/user.py. High Signal-to-Noise Ratio preserved with only 1 surgical file slice read."
        )

        final_response = """### Root Cause
The SQLAlchemy `User` model (`models/user.py`) was missing the `is_verified` column attribute, which was present in DB migration `004` and required by the Pydantic API response schema.

### Surgical Patch
```diff
--- a/models/user.py
+++ b/models/user.py
@@ -37,5 +37,6 @@ class User(Base):
     id = Column(Integer, primary_key=True)
     email = Column(String(255), unique=True, nullable=False)
+    is_verified = Column(Boolean, default=False, nullable=False)
     hashed_password = Column(String(255), nullable=False)
```"""

        # Verify anti-rumination on each thought step
        _, clean_turn1 = anti_rumination_filter(turn1_thought)
        _, clean_turn2 = anti_rumination_filter(turn2_thought)
        _, clean_turn3 = anti_rumination_filter(turn3_thought)

        messages = [
            ChatMessage(role=RoleEnum.USER, content=user_content),
            ChatMessage(role=RoleEnum.ASSISTANT, content="", thought=clean_turn1, hypothesis=bayes_hypo, tool_calls=[turn1_tool]),
            ChatMessage(role=RoleEnum.TOOL, content="", tool_outputs=[turn1_output]),
            ChatMessage(role=RoleEnum.ASSISTANT, content="", thought=clean_turn2, tool_calls=[turn2_tool]),
            ChatMessage(role=RoleEnum.TOOL, content="", tool_outputs=[turn2_output]),
            ChatMessage(role=RoleEnum.ASSISTANT, content=final_response, thought=clean_turn3),
        ]

        return ReasoningTrajectory(
            trajectory_id=f"surgical_debug_{spec.repo_name}",
            track=TrackEnum.MULTI_TURN_DEBUGGING,
            system_prompt=system_prompt,
            messages=messages,
            metadata={"localization_hops": 1, "prior_h1_probability": 0.65},
        )

    def generate_all_tracks_to_jsonl(self, output_file: str):
        logger.info(f"Generating synthetic reasoning dataset to: {output_file}")

        sys_spec = SystemDesignSpec(
            service_name="Global Telemetry Ingestion Service",
            target_rps=500000,
            p99_latency_ms=25.0,
            fault_tolerance_requirements=["Circuit Breaking", "Kafka Partition Sharding", "Multi-AZ Pod Anti-Affinity"],
            storage_architecture="ClickHouse + Redis Cluster + Kafka",
            compute_infrastructure="Kubernetes with Envoy Service Mesh",
        )

        dsa_spec = DSASpec(
            problem_title="Longest Substring with At Most K Distinct Characters",
            problem_statement="Given a string s and an integer k, return the length of the longest substring containing at most k distinct characters.",
            constraints=["1 <= s.length <= 10^5", "0 <= k <= 50"],
            brute_force_time="O(N^2)",
            target_time_complexity="O(N)",
            target_space_complexity="O(K)",
            edge_cases=["k=0", "k >= unique characters in s"],
        )

        debug_spec = DebuggingAgentSpec(
            repo_name="core-user-service",
            bug_description="Frontend 500 error on profile fetch due to unmarshalled user attribute.",
            stack_trace="File 'services/user.py', line 62, in get_profile\nKeyError: 'is_verified'",
            available_tools=["bash", "read_file", "edit_file", "grep"],
            expected_patch="Add is_verified column to models/user.py",
        )

        samples = [
            self.build_system_design_sample(sys_spec),
            self.build_dsa_sample(dsa_spec),
            self.build_tool_debugging_sample(debug_spec),
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_chatml_json(), ensure_ascii=False) + "\n")

        logger.info(f"Successfully generated {len(samples)} surgical reasoning trajectories -> {output_file}")


if __name__ == "__main__":
    pipeline = SyntheticReasoningPipeline()
    pipeline.generate_all_tracks_to_jsonl("data/processed/sft_reasoning/sample_reasoning_data.jsonl")
