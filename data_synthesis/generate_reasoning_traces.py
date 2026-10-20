"""
Synthetic Chain-of-Thought (CoT) and Agentic Data Pipeline
Formats high-grade training samples across 3 specialized tracks:
1. System Design & Fault Tolerance (High RPS, SPOF analysis, IaC/Kubernetes)
2. DSA & Asymptotic Optimization (Constraint proofs, Big-O analysis, TLE elimination)
3. Multi-Turn Tool Calling & Debugging (ReAct loop with tool_call and tool_output)
"""

import json
import os
import sys
from typing import List, Dict, Any, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_synthesis.schemas import (
    THINK_START,
    THINK_END,
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
)

logger = setup_logger("generate_reasoning_traces")


class SyntheticReasoningPipeline:
    """Generates and serializes structured reasoning trajectories for SFT."""

    def __init__(self, output_dir: str = "data/processed/sft_reasoning"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ==========================================================================
    # Track 1: System Design & Fault Tolerance Track
    # ==========================================================================
    def build_system_design_sample(self, spec: SystemDesignSpec) -> ReasoningTrajectory:
        """
        Creates a system design problem pairing high-throughput requirements with
        rigorous trade-off analysis in <think> before outputting production architecture code.
        """
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

        # Explicit Chain-of-Thought reasoning block
        thought_process = f"""1. Capacity & Bottleneck Sizing:
- Target is {spec.target_rps:,} RPS. At 1KB payload per request, ingress bandwidth is ~{spec.target_rps * 1 / 1000:.1f} MB/s (~{spec.target_rps * 8 / 1000:.1f} Gbps).
- A single PostgreSQL instance saturates at ~10,000 write RPS. Direct database writes will trigger connection pool starvation and cascading failure.
- Solution: Decouple ingestion via partitioned Kafka/Redpanda topic with at-least-once delivery guarantees.

2. Single Point of Failure (SPOF) Analysis:
- Cache Stampede: If Redis cluster nodes fail or TTLs expire simultaneously, database will be inundated.
- Solution: Probabilistic early expiration (XFetch algorithm) + Redis Cluster with multi-AZ read replicas.
- Network Partitioning (CAP Theorem): Favor Availability and Partition Tolerance (AP) for non-financial metrics, CP with Raft consensus for stateful transactions.

3. Resilience & Graceful Degradation:
- Implement client-side token bucket rate limiting + Envoy proxy adaptive concurrency limits.
- Circuit breaker: Trip when 5xx error rate exceeds 5% over 10s sliding window; serve stale cache during trip.
- Kubernetes HPA: Target 70% CPU utilization with PDB (PodDisruptionBudget) ensuring minAvailable=3 replicas."""

        assistant_code_response = f"""### Architectural Implementation: {spec.service_name}

#### 1. Resilient Go Backend with Circuit Breaking
```go
package main

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/sony/gobreaker"
)

type ResilientService struct {
	cb *gobreaker.CircuitBreaker
}

func NewResilientService() *ResilientService {
	settings := gobreaker.Settings{
		Name:        "{spec.service_name}",
		MaxRequests: 5,
		Interval:    10 * time.Second,
		Timeout:     5 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
			return counts.Requests >= 100 && failureRatio >= 0.05
		},
	}
	return &ResilientService{cb: gobreaker.NewCircuitBreaker(settings)}
}

func (s *ResilientService) Execute(ctx context.Context, fn func() (any, error)) (any, error) {
	return s.cb.Execute(func() (any, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			return fn()
		}
	})
}
```

#### 2. Kubernetes High-Availability Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {spec.service_name.lower().replace(" ", "-")}
  labels:
    tier: backend
spec:
  replicas: 8
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0
  selector:
    matchLabels:
      app: {spec.service_name.lower().replace(" ", "-")}
  template:
    metadata:
      labels:
        app: {spec.service_name.lower().replace(" ", "-")}
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: [{spec.service_name.lower().replace(" ", "-")}]
                topologyKey: "topology.kubernetes.io/zone"
      containers:
        - name: server
          image: internal-registry/{spec.service_name.lower()}:v1.2.0
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 3
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
        """
        Structures competitive programming problems into explicit reasoning phases:
        (a) Constraints & edge-cases
        (b) Brute-force failure analysis (TLE/MLE)
        (c) Optimal data structure selection
        (d) Mathematical Big-O proof
        """
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
- Constraints: {', '.join(spec.constraints)}.
- If N >= 10^5, any O(N^2) solution results in 10^10 operations, heavily exceeding the ~10^8 operations/sec budget.
- Critical Edge Cases: {', '.join(spec.edge_cases)}.

Phase 2: Brute-Force Failure Proof (TLE Analysis):
- A naive nested loop or recursion takes {spec.brute_force_time}.
- Under max input, operations count = 10^10 -> Time Limit Exceeded (TLE) guaranteed.

Phase 3: Optimal Invariant Selection:
- To achieve {spec.target_time_complexity}, we maintain an invariant over sliding window / monotonic queue / prefix tree.
- State Transition: We expand the right pointer, and shrink left pointer only when the invariant is violated.

Phase 4: Big-O Complexity Proof:
- Time: Each element is added at most once and removed at most once by pointers. Amortized 2 * N steps -> {spec.target_time_complexity}.
- Space: Auxiliary hash table / deque stores at most distinct keys -> {spec.target_space_complexity}."""

        code_solution = f"""```python
from typing import List
from collections import defaultdict

def solve(arr: List[int], k: int) -> int:
    \"\"\"
    Solves {spec.problem_title} in {spec.target_time_complexity} time and {spec.target_space_complexity} space.
    \"\"\"
    if not arr or k <= 0:
        return 0

    left = 0
    max_length = 0
    freq_map = defaultdict(int)

    for right in range(len(arr)):
        freq_map[arr[right]] += 1

        # Maintain sliding window invariant
        while len(freq_map) > k:
            freq_map[arr[left]] -= 1
            if freq_map[arr[left]] == 0:
                del freq_map[arr[left]]
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length

# Unit Verification
if __name__ == "__main__":
    assert solve([1, 2, 1, 2, 3], 2) == 4
    assert solve([], 1) == 0
    print("All algorithmic unit tests passed successfully!")
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
    # Track 3: Multi-Turn Tool Calling & Debugging Track
    # ==========================================================================
    def build_tool_debugging_sample(self, spec: DebuggingAgentSpec) -> ReasoningTrajectory:
        """
        Creates a multi-turn ReAct trajectory:
        User -> Assistant (<think> -> <tool_call>) -> Environment (<tool_output>) ->
        Assistant (<think> -> revised hypothesis -> patch).
        """
        system_prompt = (
            "You are an autonomous Software Debugging Agent with access to tools: bash, grep, "
            "read_file, edit_file. Before invoking any tool, analyze hypotheses in <think>...</think>. "
            "Inspect execution feedback inside <tool_output>...</tool_output> to correct bugs."
        )

        user_content = (
            f"Bug Report in repo '{spec.repo_name}':\n"
            f"{spec.bug_description}\n\n"
            f"Traceback:\n```\n{spec.stack_trace}\n```\n"
            "Diagnose the root cause, verify the failure, and produce a working fix."
        )

        # Turn 1: Initial hypothesis and grep invocation
        turn1_thought = (
            "The stack trace shows a RecursionError in `auth/token_validator.py` at line 42. "
            "Hypothesis: Cyclic dependency or missing base case when verifying revoked JWT tokens. "
            "Step 1: Inspect `auth/token_validator.py` around line 42 using read_file tool."
        )
        turn1_tool = ToolCall(name="read_file", arguments={"path": "auth/token_validator.py", "start_line": 35, "end_line": 55})

        turn1_output = ToolOutput(
            name="read_file",
            output="""35: def validate_token_chain(token, visited=None):
36:     if visited is None:
37:         visited = set()
38:     
39:     parent_token = token.get_parent()
40:     if parent_token:
41:         # BUG: visited set was not updated before recursive call
42:         return validate_token_chain(parent_token, visited)
43:     return True""",
            exit_code=0,
        )

        # Turn 2: Analyze tool output, invoke unit test to confirm bug
        turn2_thought = (
            "Line 42 calls `validate_token_chain` recursively, but line 41 never added `token.id` "
            "into `visited`. If a circular parent relationship exists, it triggers infinite recursion. "
            "Let's run pytest using bash tool to confirm the failure."
        )
        turn2_tool = ToolCall(name="bash", arguments={"command": "pytest tests/test_auth.py -k test_circular_token"})

        turn2_output = ToolOutput(
            name="bash",
            output="FAILED tests/test_auth.py::test_circular_token - RecursionError: maximum recursion depth exceeded",
            exit_code=1,
        )

        # Turn 3: Fix bug and verify test passes
        turn3_thought = (
            "The test failure confirms the diagnosis. Fix: Add token.id to visited before "
            "recursion; return False if token.id is already in visited. Let's output the final patch."
        )

        final_response = f"""### Root Cause Analysis
In `auth/token_validator.py`, the recursive function `validate_token_chain` did not register `token.id` into the `visited` set before recursing down the chain. When circular token hierarchies occur, this causes an infinite recursive loop culminating in a `RecursionError`.

### Git Patch
```diff
--- a/auth/token_validator.py
+++ b/auth/token_validator.py
@@ -37,6 +37,9 @@ def validate_token_chain(token, visited=None):
     if visited is None:
         visited = set()
     
+    if token.id in visited:
+        return False
+    visited.add(token.id)
+    
     parent_token = token.get_parent()
     if parent_token:
-        return validate_token_chain(parent_token, visited)
+        return validate_token_chain(parent_token, visited)
     return True
```"""

        messages = [
            ChatMessage(role=RoleEnum.USER, content=user_content),
            ChatMessage(role=RoleEnum.ASSISTANT, content="", thought=turn1_thought, tool_calls=[turn1_tool]),
            ChatMessage(role=RoleEnum.TOOL, content="", tool_outputs=[turn1_output]),
            ChatMessage(role=RoleEnum.ASSISTANT, content="", thought=turn2_thought, tool_calls=[turn2_tool]),
            ChatMessage(role=RoleEnum.TOOL, content="", tool_outputs=[turn2_output]),
            ChatMessage(role=RoleEnum.ASSISTANT, content=final_response, thought=turn3_thought),
        ]

        return ReasoningTrajectory(
            trajectory_id=f"debug_{spec.repo_name}",
            track=TrackEnum.MULTI_TURN_DEBUGGING,
            system_prompt=system_prompt,
            messages=messages,
            metadata={"tool_count": 2},
        )

    def generate_all_tracks_to_jsonl(self, output_file: str):
        """Generates representative samples across all 3 tracks and saves to disk."""
        logger.info(f"Generating synthetic reasoning dataset to: {output_file}")

        # Spec 1: System Design
        sys_spec = SystemDesignSpec(
            service_name="Global Telemetry Ingestion Service",
            target_rps=500000,
            p99_latency_ms=25.0,
            fault_tolerance_requirements=["Circuit Breaking", "Kafka Partition Sharding", "Multi-AZ Pod Anti-Affinity"],
            storage_architecture="ClickHouse + Redis Cluster + Kafka",
            compute_infrastructure="Kubernetes with Envoy Service Mesh",
        )

        # Spec 2: DSA
        dsa_spec = DSASpec(
            problem_title="Longest Substring with At Most K Distinct Characters",
            problem_statement="Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.",
            constraints=["1 <= s.length <= 10^5", "0 <= k <= 50"],
            brute_force_time="O(N^2) or O(N^3)",
            target_time_complexity="O(N)",
            target_space_complexity="O(K)",
            edge_cases=["k=0", "k >= unique characters in s", "s contains repeated characters"],
        )

        # Spec 3: Debugging ReAct
        debug_spec = DebuggingAgentSpec(
            repo_name="core-auth-gateway",
            bug_description="Infinite recursion crashes worker process during cyclical token exchange.",
            stack_trace="File 'auth/token_validator.py', line 42, in validate_token_chain\nRecursionError: maximum recursion depth exceeded",
            available_tools=["bash", "read_file", "edit_file", "grep"],
            expected_patch="Cycle detection in visited set",
        )

        samples = [
            self.build_system_design_sample(sys_spec),
            self.build_dsa_sample(dsa_spec),
            self.build_tool_debugging_sample(debug_spec),
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_chatml_json(), ensure_ascii=False) + "\n")

        logger.info(f"Successfully serialized {len(samples)} synthetic reasoning trajectories.")


if __name__ == "__main__":
    pipeline = SyntheticReasoningPipeline()
    pipeline.generate_all_tracks_to_jsonl("data/processed/sft_reasoning/sample_reasoning_data.jsonl")
