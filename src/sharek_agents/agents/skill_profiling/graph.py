import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from sharek_agents.agents.skill_profiling.prompts import SYSTEM_PROMPT
from sharek_agents.agents.skill_profiling.schemas import AgentResponse, ErrorInfo, SkillProfilingResult
from sharek_agents.common.llm import get_llm


async def _invoke_llm(
    prompt: ChatPromptTemplate, structured: object, evidence_json: str
) -> SkillProfilingResult:
    if os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower() == "groq":
        return await _invoke_groq_json(evidence_json)

    try:
        return await (prompt | structured).ainvoke({"evidence": evidence_json})
    except (ValidationError, ValueError, TypeError):
        retry_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", (
                "Your previous response did not match the required schema. "
                "Fix the format.\n\nEvidence:\n\n{evidence}"
            )),
        ])
        return await (retry_prompt | structured).ainvoke({"evidence": evidence_json})


async def run(repo_identifier: str, evidence: dict | None = None) -> AgentResponse:
    if evidence is None:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="evidence_required",
                message="No evidence provided for profiling",
                retryable=False,
            ),
        )
    evidence_json = json.dumps(evidence, indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Profile the repositories based on this evidence:\n\n{evidence}"),
    ])

    try:
        structured = (
            None
            if os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower() == "groq"
            else get_llm().with_structured_output(
                SkillProfilingResult, method="function_calling"
            )
        )
        result = await _invoke_llm(prompt, structured, evidence_json)
        return AgentResponse(
            status="success",
            skills=result.skills,
        )
    except Exception:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="llm_provider_error",
                message="LLM provider returned an error",
                retryable=True,
            ),
        )
