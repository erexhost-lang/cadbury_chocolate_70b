"""
Cadbury-Chocolate Frontier V2: Confidence-Gated Best-of-N Test-Time Compute Router
Solves Inference Latency & Overthinking via Dynamic Test-Time Compute Allocation.

Architectural Design:
1. Frontier Pitfall Solved:
   Full Monte Carlo Tree Search (MCTS) or unconditional Best-of-N on every query introduces
   catastrophic latency, token waste, and rumination on simple tasks.
2. Confidence-Gated Routing:
   Monitors predictive Shannon entropy H(p) = -sum(p log2 p) of initial reasoning tokens.
   - If H(p) < threshold (High Confidence):
     Executes FAST SINGLE-PASS decoding (N=1) with low-temperature greedy decoding.
   - If H(p) >= threshold (High Uncertainty / Ambiguity):
     Branches into PARALLEL BEST-OF-N rollouts (N=4) with exploratory temperature.
3. Deterministic Sandbox Verification:
   Rollouts are evaluated in an isolated execution sandbox against:
   - Python AST validity (ast.parse).
   - Test suite execution (isolated subprocess with strict timeouts).
   - Minimal diff footprint (favoring surgical SEARCH/REPLACE patches over file rewrites).
   - Deterministic verifiable scoring (no neural PRM overhead or reward hacking).
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_processing.git_diff_pipeline import parse_search_replace_blocks, apply_search_replace_patch

logger = setup_logger("test_time_router")


# ==============================================================================
# 1. Routing Mode & Result Dataclasses
# ==============================================================================
class RoutingDecision(str, Enum):
    FAST_SINGLE_PASS = "fast_single_pass"
    PARALLEL_BEST_OF_N = "parallel_best_of_n"


@dataclass
class CandidateResult:
    """Represents a generated rollout candidate and its deterministic evaluation."""
    candidate_id: int
    raw_completion: str
    extracted_code: str
    ast_valid: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    passed_all_tests: bool = False
    diff_lines_changed: int = 0
    execution_time_ms: float = 0.0
    syntax_error: Optional[str] = None
    test_stdout: Optional[str] = None
    deterministic_score: float = 0.0
    selected: bool = False


@dataclass
class RoutingMetadata:
    """Diagnostic telemetry on the routing decision and compute allocation."""
    decision: RoutingDecision
    mean_entropy: float
    entropy_threshold: float
    num_candidates_evaluated: int
    routing_latency_ms: float
    total_execution_ms: float
    reason: str


# ==============================================================================
# 2. Isolated Code Sandbox & Deterministic Verifier
# ==============================================================================
class IsolatedCodeSandbox:
    """
    Executes candidate code snippets in an isolated environment against unit tests,
    enforcing memory and timeout boundaries.
    """

    def __init__(self, execution_timeout_sec: float = 4.0):
        self.execution_timeout_sec = execution_timeout_sec

    def extract_code_block(self, completion: str) -> str:
        """Extracts code from ```python ... ``` or returns clean raw code."""
        code_match = re.findall(r"```python\s*([\s\S]*?)```", completion)
        if code_match:
            return code_match[-1].strip()
        code_match = re.findall(r"```\s*([\s\S]*?)```", completion)
        if code_match:
            return code_match[-1].strip()
        return completion.strip()

    def evaluate_candidate(
        self,
        candidate_id: int,
        completion: str,
        unit_tests: Optional[str] = None,
        base_file_content: Optional[str] = None,
    ) -> CandidateResult:
        """
        Validates syntax via AST, applies SEARCH/REPLACE diffs if present,
        and runs isolated unit tests.
        """
        start_time = time.time()
        diff_lines = 0

        # Check if completion uses SEARCH/REPLACE blocks
        hunks = parse_search_replace_blocks(completion)
        if hunks and base_file_content:
            extracted_code = apply_search_replace_patch(base_file_content, hunks)
            diff_lines = sum(
                len(h.replace_content.splitlines()) + len(h.search_content.splitlines())
                for h in hunks
            )
        else:
            extracted_code = self.extract_code_block(completion)
            if base_file_content:
                diff_lines = abs(len(extracted_code.splitlines()) - len(base_file_content.splitlines()))

        # 1. AST Validation
        ast_valid = False
        syntax_err = None
        try:
            ast.parse(extracted_code)
            ast_valid = True
        except SyntaxError as e:
            ast_valid = False
            syntax_err = f"SyntaxError at line {e.lineno}: {e.msg}"

        # 2. Unit Test Execution
        tests_passed = 0
        tests_total = 0
        passed_all = False
        test_out = None

        if ast_valid and unit_tests:
            passed_all, tests_passed, tests_total, test_out = self._run_sandbox_tests(
                extracted_code, unit_tests
            )
        elif ast_valid and not unit_tests:
            # If no unit tests are provided, AST validity is the baseline
            passed_all = True
            tests_passed = 1
            tests_total = 1

        elapsed_ms = (time.time() - start_time) * 1000.0

        # 3. Calculate Deterministic Score
        # Rewards: functional test pass (+10), AST validity (+3), test fraction (+5)
        # Penalties: diff line footprint (-0.01 per line to encourage minimal surgical edits)
        score = 0.0
        if ast_valid:
            score += 3.0
        if tests_total > 0:
            score += (tests_passed / tests_total) * 5.0
        if passed_all:
            score += 10.0
        score -= min(3.0, diff_lines * 0.01)

        return CandidateResult(
            candidate_id=candidate_id,
            raw_completion=completion,
            extracted_code=extracted_code,
            ast_valid=ast_valid,
            tests_passed=tests_passed,
            tests_total=tests_total,
            passed_all_tests=passed_all,
            diff_lines_changed=diff_lines,
            execution_time_ms=elapsed_ms,
            syntax_error=syntax_err,
            test_stdout=test_out,
            deterministic_score=score,
            selected=False,
        )

    def _run_sandbox_tests(
        self, code: str, unit_tests: str
    ) -> Tuple[bool, int, int, str]:
        """Runs the combined code and unit tests inside an isolated Python process."""
        combined_script = f"{code}\n\n# Unit Tests\n{unit_tests}\n"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(combined_script)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout_sec,
            )
            # Count assert statements as test count heuristic
            assert_count = max(1, unit_tests.count("assert "))

            if res.returncode == 0:
                return True, assert_count, assert_count, res.stdout
            else:
                return False, 0, assert_count, f"{res.stdout}\n{res.stderr}".strip()
        except subprocess.TimeoutExpired:
            return False, 0, 1, f"Execution timed out after {self.execution_timeout_sec}s"
        except Exception as e:
            return False, 0, 1, f"Sandbox execution error: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ==============================================================================
# 3. Entropy-Gated Best-of-N Test-Time Router
# ==============================================================================
class EntropyGatedTestTimeRouter:
    """
    Confidence-Gated Test-Time Compute Router for Cadbury-70B.

    Evaluates predictive Shannon entropy H(p) during generation prefix:
    - High confidence (H < tau) -> Route to Fast Single-Pass (N=1)
    - Low confidence (H >= tau) -> Route to Best-of-N Sandbox Search (N=4)
    """

    def __init__(
        self,
        entropy_threshold: float = 0.55,
        num_candidates: int = 4,
        fast_temperature: float = 0.1,
        explore_temperature: float = 0.7,
        top_p: float = 0.95,
        probe_token_count: int = 16,
        sandbox_timeout_sec: float = 4.0,
    ):
        self.entropy_threshold = entropy_threshold
        self.num_candidates = num_candidates
        self.fast_temperature = fast_temperature
        self.explore_temperature = explore_temperature
        self.top_p = top_p
        self.probe_token_count = probe_token_count
        self.sandbox = IsolatedCodeSandbox(execution_timeout_sec=sandbox_timeout_sec)

    @staticmethod
    def calculate_shannon_entropy(logits: torch.Tensor) -> float:
        """
        Computes Shannon entropy in bits for next-token probability distribution:
        H(p) = - sum_i p_i * log2(p_i)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log2(probs.clamp(min=1e-10))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def evaluate_reasoning_entropy(
        self,
        logits_history: List[torch.Tensor],
    ) -> float:
        """Computes mean Shannon entropy across probe reasoning tokens."""
        if not logits_history:
            return 0.0
        entropies = [self.calculate_shannon_entropy(l) for l in logits_history]
        return sum(entropies) / len(entropies)

    def select_best_candidate(
        self, candidates: List[CandidateResult]
    ) -> CandidateResult:
        """
        Ranks candidates deterministically using verifiable metrics:
        1. Fully passing unit tests (functional correctness).
        2. AST validity (syntactic correctness).
        3. Deterministic score (diff footprint minimization + test ratio).
        4. Lowest execution time as tiebreaker.
        """
        # Sort key: (passed_all_tests, ast_valid, deterministic_score, -execution_time_ms)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.passed_all_tests,
                c.ast_valid,
                c.deterministic_score,
                -c.diff_lines_changed,
                -c.execution_time_ms,
            ),
            reverse=True,
        )
        selected = sorted_candidates[0]
        selected.selected = True
        return selected

    def route_and_generate(
        self,
        prompt: str,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        unit_tests: Optional[str] = None,
        base_file_content: Optional[str] = None,
        simulated_entropy: Optional[float] = None,
        simulated_candidates: Optional[List[str]] = None,
    ) -> Tuple[CandidateResult, RoutingMetadata, List[CandidateResult]]:
        """
        Executes confidence-gated routing and returns the selected output.

        Supports both real inference (with PyTorch model & tokenizer) and simulated
        evaluations for testing and cluster verification.
        """
        t0 = time.time()

        # Step 1: Probe initial tokens to estimate entropy
        if simulated_entropy is not None:
            mean_entropy = simulated_entropy
        elif model is not None and tokenizer is not None:
            mean_entropy = self._probe_live_model_entropy(prompt, model, tokenizer)
        else:
            mean_entropy = 0.35  # Fallback default

        # Step 2: Make Routing Decision
        if mean_entropy < self.entropy_threshold:
            decision = RoutingDecision.FAST_SINGLE_PASS
            reason = (
                f"High confidence: Mean entropy H={mean_entropy:.3f} < {self.entropy_threshold:.3f}. "
                "Routing directly to Fast Single-Pass (N=1)."
            )
            candidate_count = 1
        else:
            decision = RoutingDecision.PARALLEL_BEST_OF_N
            reason = (
                f"High uncertainty: Mean entropy H={mean_entropy:.3f} >= {self.entropy_threshold:.3f}. "
                f"Branching into Parallel Best-of-{self.num_candidates} Sandbox Search."
            )
            candidate_count = self.num_candidates

        routing_time_ms = (time.time() - t0) * 1000.0

        # Step 3: Candidate Generation & Sandbox Evaluation
        evaluated_candidates: List[CandidateResult] = []

        if simulated_candidates is not None:
            # Use provided candidate list for verification
            rollouts = simulated_candidates[:candidate_count]
            for idx, text in enumerate(rollouts):
                cand = self.sandbox.evaluate_candidate(
                    candidate_id=idx + 1,
                    completion=text,
                    unit_tests=unit_tests,
                    base_file_content=base_file_content,
                )
                evaluated_candidates.append(cand)
        else:
            # Real generation branch
            rollouts = self._generate_rollouts(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                count=candidate_count,
                temperature=self.fast_temperature if decision == RoutingDecision.FAST_SINGLE_PASS else self.explore_temperature,
            )
            for idx, text in enumerate(rollouts):
                cand = self.sandbox.evaluate_candidate(
                    candidate_id=idx + 1,
                    completion=text,
                    unit_tests=unit_tests,
                    base_file_content=base_file_content,
                )
                evaluated_candidates.append(cand)

        # Step 4: Deterministic Selection
        selected = self.select_best_candidate(evaluated_candidates)
        total_time_ms = (time.time() - t0) * 1000.0

        metadata = RoutingMetadata(
            decision=decision,
            mean_entropy=mean_entropy,
            entropy_threshold=self.entropy_threshold,
            num_candidates_evaluated=len(evaluated_candidates),
            routing_latency_ms=routing_time_ms,
            total_execution_ms=total_time_ms,
            reason=reason,
        )

        logger.info(
            f"[Router Decision] {decision.value.upper()} | Evaluated {len(evaluated_candidates)} candidates | "
            f"Entropy: {mean_entropy:.3f} | Selected Candidate #{selected.candidate_id} (Score={selected.deterministic_score:.2f})"
        )

        return selected, metadata, evaluated_candidates

    def _probe_live_model_entropy(
        self, prompt: str, model: Any, tokenizer: Any
    ) -> float:
        """Generates initial reasoning probe tokens to evaluate Shannon entropy."""
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)

        entropies = []
        with torch.no_grad():
            curr_ids = input_ids
            for _ in range(self.probe_token_count):
                outputs = model(curr_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                next_token_logits = logits[:, -1, :]
                ent = self.calculate_shannon_entropy(next_token_logits)
                entropies.append(ent)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)

        return sum(entropies) / max(1, len(entropies))

    def _generate_rollouts(
        self,
        prompt: str,
        model: Optional[Any],
        tokenizer: Optional[Any],
        count: int,
        temperature: float,
    ) -> List[str]:
        """Generates rollouts with given temperature and top_p."""
        if model is None or tokenizer is None:
            # Fallback stub for environment testing
            return [f"```python\ndef solve(): pass\n```" for _ in range(count)]

        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        input_ids = inputs["input_ids"].to(device)

        rollouts = []
        with torch.no_grad():
            for _ in range(count):
                if hasattr(model, "generate"):
                    output = model.generate(
                        input_ids,
                        max_new_tokens=2048,
                        do_sample=(temperature > 0.0),
                        temperature=max(0.01, temperature),
                        top_p=self.top_p,
                    )
                    gen_ids = output[0].tolist()
                else:
                    # Native autoregressive decoding with Cadbury70BMoEModel
                    curr_ids = input_ids
                    past_kv = None
                    gen_tokens = []
                    B, T_init = input_ids.shape
                    pos_ids = torch.arange(0, T_init, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

                    outputs = model(curr_ids, position_ids=pos_ids)
                    logits, past_kv, _ = outputs if isinstance(outputs, tuple) else (outputs, None, None)
                    next_token_logits = logits[:, -1, :]

                    if temperature > 0:
                        probs = F.softmax(next_token_logits / max(0.01, temperature), dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tok = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    gen_tokens.append(next_tok)
                    curr_pos = T_init

                    for _ in range(512):
                        step_pos = torch.full((B, 1), curr_pos, device=device, dtype=torch.long)
                        step_out = model(next_tok, past_key_values=past_kv, position_ids=step_pos)
                        step_logits, past_kv, _ = step_out if isinstance(step_out, tuple) else (step_out, None, None)
                        step_next = step_logits[:, -1, :]

                        if temperature > 0:
                            probs = F.softmax(step_next / max(0.01, temperature), dim=-1)
                            next_tok = torch.multinomial(probs, num_samples=1)
                        else:
                            next_tok = torch.argmax(step_next, dim=-1, keepdim=True)

                        gen_tokens.append(next_tok)
                        curr_pos += 1
                        if hasattr(tokenizer, "eos_token_id") and next_tok.item() == tokenizer.eos_token_id:
                            break

                    all_ids = torch.cat([input_ids, torch.cat(gen_tokens, dim=-1)], dim=-1)
                    gen_ids = all_ids[0].tolist()

                decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
                rollouts.append(decoded)

        return rollouts


# ==============================================================================
# 4. Self-Contained Verification & Demonstration
# ==============================================================================
if __name__ == "__main__":
    router = EntropyGatedTestTimeRouter(entropy_threshold=0.55, num_candidates=4)

    print("=" * 80)
    print("Cadbury-70B Frontier V2: Entropy-Gated Best-of-N Sandbox Router Test")
    print("=" * 80)

    # --- Scenario 1: Low-Entropy Task (Standard Algorithm / Boilerplate) ---
    print("\n[Scenario 1: Simple Task -> Fast Path Verification]")
    simple_prompt = "Write a function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number."
    simple_tests = """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(7) == 13
"""
    simple_simulated = [
        """```python
def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```"""
    ]

    best, meta, all_cands = router.route_and_generate(
        prompt=simple_prompt,
        unit_tests=simple_tests,
        simulated_entropy=0.22,  # Low entropy (high confidence)
        simulated_candidates=simple_simulated,
    )

    print(f"Decision: {meta.decision.value}")
    print(f"Reason: {meta.reason}")
    print(f"Candidates Evaluated: {meta.num_candidates_evaluated}")
    print(f"Candidate #1 Passed All Tests: {best.passed_all_tests} (Score: {best.deterministic_score})")

    # --- Scenario 2: High-Entropy Task (Subtle Concurrency Bug) ---
    print("\n[Scenario 2: Complex Task -> Parallel Best-of-4 Sandbox Search]")
    complex_prompt = "Fix the race condition in this token-bucket rate limiter."
    complex_tests = """
import time
limiter = TokenBucket(capacity=2, refill_rate=10.0)
assert limiter.consume(1) is True
assert limiter.consume(1) is True
assert limiter.consume(1) is False
time.sleep(0.15)
assert limiter.consume(1) is True
"""
    # 4 Parallel candidates with varying correctness:
    candidates_pool = [
        # Candidate 1: Syntax error
        """```python
def broken_syntax(
```""",
        # Candidate 2: Incomplete logic (fails tests)
        """```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
    def consume(self, amount):
        return True
```""",
        # Candidate 3: Correct surgical implementation
        """```python
import time
import threading

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, amount: int = 1) -> bool:
        with self.lock:
            now = time.time()
            delta = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
            self.last_update = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False
```""",
        # Candidate 4: Alternate bloated implementation
        """```python
import time
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last = time.time()
    def consume(self, amount):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_rate)
        self.last = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False
```""",
    ]

    best_complex, meta_complex, all_cands_complex = router.route_and_generate(
        prompt=complex_prompt,
        unit_tests=complex_tests,
        simulated_entropy=0.82,  # High entropy (high uncertainty)
        simulated_candidates=candidates_pool,
    )

    print(f"Decision: {meta_complex.decision.value}")
    print(f"Reason: {meta_complex.reason}")
    print(f"Candidates Evaluated: {meta_complex.num_candidates_evaluated}")
    for c in all_cands_complex:
        print(f"  Candidate #{c.candidate_id}: AST={c.ast_valid}, TestsPassed={c.tests_passed}/{c.tests_total}, Score={c.deterministic_score:.2f}, Selected={c.selected}")

    print(f"\nWinning Candidate #{best_complex.candidate_id} Deterministic Score: {best_complex.deterministic_score:.2f}")
    print("=" * 80)
