"""
Data Schemas for Chain-of-Thought (CoT) and Agentic Tool Interaction
Defines ChatML formats and structured special tokens for 70B reasoning training:
- <think>...</think> for deliberate algorithmic reflection and architectural trade-offs
- <tool_call>...</tool_call> for structured environment actions (ReAct)
- <tool_output>...</tool_output> for sandbox execution feedback and stack traces
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Explicit Control Tokens for Reasoning & Tool Calling
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_OUTPUT_START = "<tool_output>"
TOOL_OUTPUT_END = "</tool_output>"


class RoleEnum(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TrackEnum(str, Enum):
    SYSTEM_DESIGN = "system_design_fault_tolerance"
    DSA_OPTIMIZATION = "dsa_and_optimization"
    MULTI_TURN_DEBUGGING = "multi_turn_tool_debugging"


class ToolCall(BaseModel):
    name: str = Field(..., description="Target tool name, e.g. 'bash', 'read_file', 'grep'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool parameter dictionary")

    def to_xml_format(self) -> str:
        """Serialize tool call to model-readable XML format."""
        import json
        args_str = json.dumps(self.arguments, ensure_ascii=False)
        return f'{TOOL_CALL_START}\n{{"name": "{self.name}", "arguments": {args_str}}}\n{TOOL_CALL_END}'


class ToolOutput(BaseModel):
    call_id: Optional[str] = None
    name: str
    output: str
    exit_code: int = 0

    def to_xml_format(self) -> str:
        """Serialize tool output to masked environment format."""
        return (
            f"{TOOL_OUTPUT_START}\n"
            f"[Tool: {self.name} | Exit Code: {self.exit_code}]\n"
            f"{self.output.strip()}\n"
            f"{TOOL_OUTPUT_END}"
        )


class ChatMessage(BaseModel):
    role: RoleEnum
    content: str
    thought: Optional[str] = Field(
        None, description="Explicit internal reasoning trace placed inside <think>...</think>"
    )
    tool_calls: Optional[List[ToolCall]] = None
    tool_outputs: Optional[List[ToolOutput]] = None

    def render_content_with_tags(self) -> str:
        """Renders the message body incorporating <think> and tool tags in canonical order."""
        parts = []
        if self.thought and self.thought.strip():
            parts.append(f"{THINK_START}\n{self.thought.strip()}\n{THINK_END}")

        if self.tool_calls:
            for tc in self.tool_calls:
                parts.append(tc.to_xml_format())

        if self.tool_outputs:
            for to in self.tool_outputs:
                parts.append(to.to_xml_format())

        if self.content and self.content.strip():
            parts.append(self.content.strip())

        return "\n\n".join(parts)


class ReasoningTrajectory(BaseModel):
    trajectory_id: str
    track: TrackEnum
    system_prompt: str
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_chatml_json(self) -> Dict[str, Any]:
        """Convert trajectory to Hugging Face standard conversation format with XML-tagged assistant responses."""
        formatted_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in self.messages:
            formatted_messages.append({
                "role": msg.role.value,
                "content": msg.render_content_with_tags()
            })

        return {
            "id": self.trajectory_id,
            "track": self.track.value,
            "conversations": formatted_messages,
            "metadata": self.metadata,
        }


# ==============================================================================
# Domain Specifications for Synthetic Generators
# ==============================================================================

class SystemDesignSpec(BaseModel):
    service_name: str
    target_rps: int
    p99_latency_ms: float
    fault_tolerance_requirements: List[str]
    storage_architecture: str
    compute_infrastructure: str  # e.g., "Kubernetes with HPA and Cilium eBPF"


class DSASpec(BaseModel):
    problem_title: str
    problem_statement: str
    constraints: List[str]
    brute_force_time: str
    target_time_complexity: str
    target_space_complexity: str
    edge_cases: List[str]


class DebuggingAgentSpec(BaseModel):
    repo_name: str
    bug_description: str
    stack_trace: str
    available_tools: List[str]
    expected_patch: str
