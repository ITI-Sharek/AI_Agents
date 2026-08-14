"""New Skill Profiling Agent package (Phase 2: ReAct core, Phase 7: MCP-ready).

This package is intentionally isolated from the existing
``skill_profiling`` workflow. It is the home of the MCP-integrated
agent; the generic ReAct core with a deterministic test tool exists, the
MCP client boundary is reserved in ``mcp_client.py``, and — since Phase
15 — a request-scoped evidence bundle plus deterministic skill-profile
construction live in ``evidence.py`` and ``skills.py``. Since Phase 16
the production service path registers the deterministic
``DetectFrameworksTool`` (PAT-scoped) and the endpoint resolves an
injectable MCP configuration.
"""

from sharek_agents.agents.skill_profiling_agent.agent import (
    AgentConfig,
    AgentProviderError,
    AgentTimeoutError,
    SkillProfilingAgent,
)
from sharek_agents.agents.skill_profiling_agent.detection import (
    TECHNOLOGY_CATEGORIES,
    Detection,
    DetectionReport,
    FrameworksDetector,
    GitHubClient,
    RepositoryAccessError,
    RepositoryContext,
    TechnologyEntry,
    TechnologyRegistry,
)
from sharek_agents.agents.skill_profiling_agent.endpoint import resolve_mcp_config
from sharek_agents.agents.skill_profiling_agent.evidence import (
    EVIDENCE_ID_PREFIX,
    EvidenceBundle,
    EvidenceRecord,
)
from sharek_agents.agents.skill_profiling_agent.mcp_client import (
    MCPClient,
    MCPClientConfig,
    MCPClientError,
    MCPConnectionError,
    MCPToolDefinition,
    MCPToolExecutionError,
    MCPToolResult,
    SkillProfilingMCPClient,
)
from sharek_agents.agents.skill_profiling_agent.schemas import (
    SkillProfileAgentOutput,
    SkillProfileAgentResponse,
    ToolActivity,
)
from sharek_agents.agents.skill_profiling_agent.service import (
    SkillProfileAgentError,
    SkillProfileAgentTimeout,
    generate_skill_profile_agent,
)
from sharek_agents.agents.skill_profiling_agent.skills import (
    MAX_SKILLS,
    PROFICIENCY_LEVELS,
    build_skill_profile,
)
from sharek_agents.agents.skill_profiling_agent.tools import (
    DetectFrameworksTool,
    GetAgentContextTool,
    McpToolAdapter,
    NativeToolCall,
    Tool,
    ToolDefinition,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
    build_mcp_tool_adapters,
)

__all__ = [
    "EVIDENCE_ID_PREFIX",
    "MAX_SKILLS",
    "PROFICIENCY_LEVELS",
    "TECHNOLOGY_CATEGORIES",
    "AgentConfig",
    "AgentProviderError",
    "AgentTimeoutError",
    "DetectFrameworksTool",
    "Detection",
    "DetectionReport",
    "EvidenceBundle",
    "EvidenceRecord",
    "FrameworksDetector",
    "GetAgentContextTool",
    "GitHubClient",
    "MCPClient",
    "MCPClientConfig",
    "MCPClientError",
    "MCPConnectionError",
    "MCPToolDefinition",
    "MCPToolExecutionError",
    "MCPToolResult",
    "McpToolAdapter",
    "NativeToolCall",
    "RepositoryAccessError",
    "RepositoryContext",
    "SkillProfileAgentError",
    "SkillProfileAgentOutput",
    "SkillProfileAgentResponse",
    "SkillProfileAgentTimeout",
    "SkillProfilingAgent",
    "SkillProfilingMCPClient",
    "TechnologyEntry",
    "TechnologyRegistry",
    "Tool",
    "ToolActivity",
    "ToolDefinition",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "build_mcp_tool_adapters",
    "build_skill_profile",
    "generate_skill_profile_agent",
    "resolve_mcp_config",
]
