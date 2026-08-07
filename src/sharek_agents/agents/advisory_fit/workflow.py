from __future__ import annotations

import asyncio
import json
import re
from typing import TypeVar

from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from sharek_agents.agents.advisory_fit import agent as agent_module
from sharek_agents.agents.advisory_fit import service as service_module
from sharek_agents.agents.advisory_fit import stages
from sharek_agents.agents.advisory_fit.prompts import (
    APPROACH_ANALYSIS_HUMAN_PROMPT,
    APPROACH_ANALYSIS_SYSTEM_PROMPT,
    UNDERSTAND_APPROACH_HUMAN_PROMPT,
    UNDERSTAND_APPROACH_SYSTEM_PROMPT,
)
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
    ApproachAnalysis,
    Assessment,
    Evidence,
    EvidenceUnderstanding,
    LevelMatch,
)
from sharek_agents.agents.advisory_fit.scoring import (
    calculate_fit_percentage,
    calculate_level_match,
)
from sharek_agents.agents.advisory_fit.state import AgentState, SkillVerificationEntry
from sharek_agents.common.llm import get_llm
from sharek_agents.config import settings


_MAX_UNDERSTANDING_ITEMS = 50
_MAX_APPROACH_ITEMS = 50

_StructuredModel = TypeVar("_StructuredModel", bound=BaseModel)


async def _structured_extraction_llm(
    system_prompt: str,
    human_prompt: str,
    schema: type[_StructuredModel],
    input_data: AdvisoryFitInput,
) -> _StructuredModel:
    """Run one bounded structured LLM extraction owned by a workflow node.

    Only the project context (title/description), the Contributor Approach
    (the primary subject), and the Contributor Evidence (supporting context)
    are supplied: the Approach and Evidence are serialized as opaque,
    untrusted data and the prompt treats them as data only. The structured
    output is validated against ``schema`` and the call is bounded by the
    configured timeout.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    structured = get_llm().with_structured_output(schema)
    result = await asyncio.wait_for(
        (prompt | structured).ainvoke(
            {
                "project_context": json.dumps(
                    {
                        "title": input_data.project.title,
                        "description": input_data.project.description,
                    },
                    indent=2,
                ),
                "contributor_approach": input_data.contributor.approach
                or "(not provided)",
                "contributor_evidence": service_module._serialize_evidence(
                    input_data.contributor.evidence
                ),
            }
        ),
        timeout=settings.ai_skill_profile_timeout_seconds,
    )
    return schema.model_validate(result)


async def _invoke_understand_approach_llm(
    input_data: AdvisoryFitInput,
) -> EvidenceUnderstanding:
    """Evidence-grounded understanding of the Contributor Approach.

    ``_structured_extraction_llm`` with the understand-approach prompt pair and
    the ``EvidenceUnderstanding`` schema. The extraction is requirement-agnostic
    and never compares against project requirements.
    """
    return await _structured_extraction_llm(
        UNDERSTAND_APPROACH_SYSTEM_PROMPT,
        UNDERSTAND_APPROACH_HUMAN_PROMPT,
        EvidenceUnderstanding,
        input_data,
    )


async def _invoke_approach_analysis_llm(
    input_data: AdvisoryFitInput,
) -> ApproachAnalysis:
    """Understanding of what the contributor intends to build.

    ``_structured_extraction_llm`` with the ``APPROACH_ANALYSIS_*`` prompt pair
    and the ``ApproachAnalysis`` schema. The extraction describes ONLY the
    intended work and never compares against project requirements or inspects
    contributor skills.
    """
    return await _structured_extraction_llm(
        APPROACH_ANALYSIS_SYSTEM_PROMPT,
        APPROACH_ANALYSIS_HUMAN_PROMPT,
        ApproachAnalysis,
        input_data,
    )


def _check_capped(items: list, context: str, label: str, max_items: int) -> None:
    """Fail-closed cap check shared by the extraction validators."""
    if len(items) > max_items:
        raise service_module.AdvisoryFitProviderError(
            f"{context} contains too many {label}: "
            f"{len(items)} (max {max_items})"
        )


def _validate_evidence_understanding(
    understanding: EvidenceUnderstanding,
    evidence: list[Evidence],
) -> None:
    """Deterministic anti-hallucination checks on the LLM extraction.

    Fail-closed: every evidence reference must point at a real evidence
    record, extracted names must be unique (case-insensitive), and item
    counts are capped. Nothing here compares against project requirements,
    verifies contributor skills, or performs scoring.
    """
    evidence_count = len(evidence)

    def check_indexes(indexes: list[int], context: str) -> None:
        out_of_range = [i for i in indexes if i < 0 or i >= evidence_count]
        if out_of_range:
            raise service_module.AdvisoryFitProviderError(
                "evidence understanding references out-of-range "
                f"{context}: {sorted(out_of_range)}"
            )

    def check_unique(items: list, key, label: str) -> None:
        seen: set[str] = set()
        for item in items:
            normalized = key(item)
            if normalized in seen:
                raise service_module.AdvisoryFitProviderError(
                    f"duplicate {label} in evidence understanding: "
                    f"'{normalized}'"
                )
            seen.add(normalized)

    for artifact in understanding.built_artifacts:
        check_indexes(artifact.evidence_indexes, "artifact evidence indexes")
    for capability in understanding.capabilities:
        check_indexes(
            capability.evidence_indexes, "capability evidence indexes"
        )
    for pattern in understanding.architectural_patterns:
        check_indexes(pattern.evidence_indexes, "pattern evidence indexes")
    for technology in understanding.technologies:
        check_indexes(technology.evidence_indexes, "technology evidence indexes")
    for experience in understanding.supported_experience:
        check_indexes(
            experience.evidence_indexes, "experience evidence indexes"
        )

    for label, items in (
        ("built artifacts", understanding.built_artifacts),
        ("capabilities", understanding.capabilities),
        ("architectural patterns", understanding.architectural_patterns),
        ("technologies", understanding.technologies),
        ("supported experience", understanding.supported_experience),
    ):
        _check_capped(
            items, "evidence understanding", label, _MAX_UNDERSTANDING_ITEMS
        )

    check_unique(
        understanding.built_artifacts,
        lambda a: a.title.casefold(),
        "built artifact",
    )
    check_unique(
        understanding.capabilities,
        lambda c: c.capability.casefold(),
        "capability",
    )
    check_unique(
        understanding.architectural_patterns,
        lambda p: p.pattern.casefold(),
        "architectural pattern",
    )
    check_unique(
        understanding.technologies,
        lambda t: t.name.casefold(),
        "technology",
    )
    check_unique(
        understanding.supported_experience,
        lambda e: e.experience.casefold(),
        "supported experience",
    )


def _validate_approach_analysis(analysis: ApproachAnalysis) -> None:
    """Deterministic anti-hallucination checks on the approach extraction.

    Fail-closed: per-category item counts are capped; item uniqueness,
    non-emptiness, and item length are enforced by the schema. Nothing here
    inspects project requirements, verifies contributor skills, or computes
    scores.
    """
    for label, items in (
        ("intended features", analysis.intended_features),
        ("intended capabilities", analysis.intended_capabilities),
        ("intended architecture", analysis.intended_architecture),
        ("intended technologies", analysis.intended_technologies),
        ("implementation plan steps", analysis.implementation_plan),
    ):
        _check_capped(
            items, "approach analysis", label, _MAX_APPROACH_ITEMS
        )


async def understand_approach(state: AgentState) -> dict:
    """Understand the Contributor Approach — the primary workflow artifact.

    Reads: ``request``.
    Writes: ``approach_analysis``, ``intended_approach``.

    The node is responsible ONLY for understanding the contributor approach.
    Two bounded structured LLM calls run here, each requirement-agnostic:

    - ``EvidenceUnderstanding`` (``approach_analysis``): what the contributor
      has built, grounded in the Evidence, with the Approach as context.
    - ``ApproachAnalysis`` (``intended_approach``): what the contributor
      intends to build, from the Approach as the primary subject and the
      Evidence strictly as supporting context.

    Both extractions are validated deterministically (evidence reference
    bounds, uniqueness, caps) and never compare project requirements, inspect
    contributor skills, generate a roadmap, or perform requirement matching.
    An empty Contributor Approach yields an empty ``ApproachAnalysis``
    without an LLM call.
    """
    request = state["request"]
    evidence = request.contributor.evidence

    if evidence:
        try:
            understanding = await _invoke_understand_approach_llm(request)
        except ValidationError as exc:
            raise service_module.AdvisoryFitProviderError(
                f"Advisory Fit provider returned invalid output: {exc}"
            ) from exc
        _validate_evidence_understanding(understanding, evidence)
    else:
        understanding = EvidenceUnderstanding()

    if request.contributor.approach.strip():
        try:
            analysis = await _invoke_approach_analysis_llm(request)
        except ValidationError as exc:
            raise service_module.AdvisoryFitProviderError(
                f"Advisory Fit provider returned invalid output: {exc}"
            ) from exc
        _validate_approach_analysis(analysis)
    else:
        analysis = ApproachAnalysis()

    return {"approach_analysis": understanding, "intended_approach": analysis}


_WORD_RE = re.compile(r"[a-z0-9]+")


def _intended_approach_corpus(analysis: ApproachAnalysis) -> str:
    """Flatten the intended-approach analysis into searchable text.

    Uses ONLY the structured categories of ``ApproachAnalysis`` — intended
    features, capabilities, architecture, technologies, and implementation
    plan steps; the narrative ``summary`` is deliberately excluded so
    selection is driven strictly by the itemized, intended-work analysis of
    the approach.
    """
    parts: list[str] = []
    parts += analysis.intended_features
    parts += analysis.intended_capabilities
    parts += analysis.intended_architecture
    parts += analysis.intended_technologies
    parts += analysis.implementation_plan
    return " ".join(parts)


def _phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text.casefold()) is not None


def _classify_approach_relevance(
    skill: str,
    corpus: str,
) -> tuple[str, bool]:
    """Classify one requirement against the intended-approach analysis only.

    Returns ``(decision, relevant)`` where ``decision`` is ``DIRECT``,
    ``PARTIAL``, or ``NOT_MENTIONED`` and ``relevant`` is ``True`` for the
    first two. No contributor skills, raw evidence, fit scores, or
    requirement matching are involved.
    """
    key = skill.casefold()
    tokens = _WORD_RE.findall(key)

    if _phrase_present(corpus, key):
        return "DIRECT", True

    if len(tokens) > 1:
        meaningful = [t for t in tokens if len(t) >= 4]
        if meaningful and any(_phrase_present(corpus, t) for t in meaningful):
            return "PARTIAL", True

    for word in _WORD_RE.findall(corpus):
        if len(word) < 4:
            continue
        if key in word or word in key:
            return "PARTIAL", True

    return "NOT_MENTIONED", False


def select_relevant_requirements(state: AgentState) -> dict:
    """Select project requirements from the intended-approach analysis only.

    Reads: ``request``, ``intended_approach``.
    Writes: ``relevant_requirements``, ``partially_relevant_requirements``,
    ``requirement_classifications``.

    Deterministic relevance classification driven ONLY by the intended
    approach (the ``ApproachAnalysis`` written as ``intended_approach`` by
    ``understand_approach``). Each project requirement is classified
    ``DIRECT``, ``PARTIAL``, or ``NOT_MENTIONED`` against ONLY the
    structured categories of that analysis — intended features,
    capabilities, architecture, technologies, and implementation plan steps
    (the narrative ``summary`` is excluded). Contributor skills, contributor
    evidence, project context, and any matching/fit results are never
    inspected; no skill verification, fit calculation, or scoring happens
    here, and no LLM call is made.

    This node is the ONLY producer of the per-requirement relevance
    classification. It writes, for EVERY project requirement, an explicit
    ``requirement_classifications`` mapping (normalized requirement name to
    ``DIRECT`` / ``PARTIAL`` / ``NOT_MENTIONED``) so downstream nodes consume
    the decision verbatim instead of reconstructing it. The ordered
    partitions ``relevant_requirements`` (DIRECT or PARTIAL, in request
    order) and ``partially_relevant_requirements`` (the PARTIAL subset) are
    kept for the matching node, which consumes them unchanged.
    """
    if "intended_approach" not in state:
        raise service_module.AdvisoryFitProviderError(
            "select_relevant_requirements requires understand_approach to "
            "have produced an analysis first"
        )
    request = state["request"]
    corpus = _intended_approach_corpus(state["intended_approach"])

    relevant_requirements: list[str] = []
    partially_relevant_requirements: list[str] = []
    requirement_classifications: dict[str, str] = {}

    for requirement in request.project_requirements:
        decision, is_relevant = _classify_approach_relevance(
            requirement.skill, corpus
        )
        requirement_classifications[requirement.skill.casefold()] = decision
        if decision == "PARTIAL":
            partially_relevant_requirements.append(requirement.skill)
        if is_relevant:
            relevant_requirements.append(requirement.skill)

    return {
        "relevant_requirements": relevant_requirements,
        "partially_relevant_requirements": partially_relevant_requirements,
        "requirement_classifications": requirement_classifications,
    }


def _declared_skill_map(skills) -> dict[str, str]:
    """Map casefolded skill -> declared level from ``contributor.skills``."""
    return {s.skill.casefold(): s.level for s in skills}


def _evidence_demonstrates(
    skill: str,
    evidence: list[Evidence],
) -> bool:
    """Whether the contributor evidence demonstrates the skill.

    A word-boundary phrase match against the skill name in any evidence
    record's title, summary, description, or technologies counts as
    demonstrated experience.
    """
    key = skill.casefold()
    for item in evidence:
        fields = [item.title, item.summary, item.description]
        fields += list(item.technologies)
        if any(f and _phrase_present(f, key) for f in fields):
            return True
    return False


def match_skills_and_evidence(state: AgentState) -> dict:
    """Verify the contributor can execute the selected requirements only.

    Reads: ``relevant_requirements``, ``request`` (``contributor.skills``,
    ``contributor.evidence``, and the required levels of
    ``project_requirements``).
    Writes: ``skill_verification``, ``level_evaluations``.

    The node is responsible ONLY for deterministic skills & evidence
    verification of the selection produced by ``select_relevant_requirements``.
    For every relevant requirement it (1) checks whether the contributor
    declared the skill in ``contributor.skills``, (2) compares the declared
    level against the required level, (3) searches ``contributor.evidence``
    for support, and (4) records the deterministic matching information:
    a declared skill is ``MATCHED``, otherwise ``NOT_EVIDENCED``.

    Requirement selection and fit calculation are out of scope: the node
    never determines relevance, never interprets the Contributor Approach,
    never reads the project description or title, and never computes a
    percentage. The output holds only the deterministic matching data
    ``calculate_fit`` consumes: no relevance values, no fit scores, no
    recommendations.
    """
    if "relevant_requirements" not in state:
        raise service_module.AdvisoryFitProviderError(
            "match_skills_and_evidence requires select_relevant_requirements "
            "to have produced a selection first"
        )
    request = state["request"]
    relevant = state["relevant_requirements"]

    declared_levels = _declared_skill_map(request.contributor.skills)
    evidence = request.contributor.evidence
    required_levels = {
        req.skill.casefold(): req.level for req in request.project_requirements
    }

    skill_verification: dict[str, SkillVerificationEntry] = {}
    level_evaluations: dict[str, LevelMatch] = {}

    for skill in relevant:
        key = skill.casefold()
        required_level = required_levels[key]
        declared_level = declared_levels.get(key)
        demonstrated = _evidence_demonstrates(skill, evidence)

        skill_verification[key] = SkillVerificationEntry(
            contributor_level=declared_level,
            skill_match=(
                "MATCHED" if declared_level is not None else "NOT_EVIDENCED"
            ),
            explanation=f"Declared knowledge: {'yes' if declared_level else 'no'}; "
            f"demonstrated experience: {'yes' if demonstrated else 'no'}.",
        )
        level_evaluations[key] = calculate_level_match(
            required_level, declared_level
        )

    return {
        "skill_verification": skill_verification,
        "level_evaluations": level_evaluations,
    }


_NOT_SELECTED_EXPLANATION = (
    "Requirement not selected: no skills or evidence matching was performed."
)


def calculate_fit(state: AgentState) -> dict:
    """Generate the final Advisory Fit — the pure aggregation stage.

    Reads: ``requirement_classifications``, ``skill_verification``,
    ``level_evaluations``, and ``request`` (``project_requirements`` only,
    for the per-requirement enumeration and required levels).
    Writes: ``requirement_assessments``, ``fit_percentage``, ``summary``,
    ``final_result``.

    This node is the ONLY node responsible for producing the final Advisory
    Fit result and is a pure aggregation stage: it combines the previous
    nodes' deterministic outputs into one ``Assessment`` per project
    requirement, in request order. The schema-required ``approach_relevance``
    is read verbatim from the ``requirement_classifications`` mapping written
    by ``select_relevant_requirements`` — that node is the only producer of
    the relevance classification, and this node never infers, reconstructs,
    classifies, or decides relevance. The matching data
    (``skill_verification``, ``level_evaluations``) is attached for every
    requirement; requirements with no matching data are reported with
    ``NOT_EVIDENCED``/``MISSING`` and a fixed explanation. The fit percentage
    reuses the existing ``calculate_fit_percentage`` scoring with unchanged
    weights, and the public result reuses the existing ``AdvisoryFitResult``
    response schema. No LLM reasoning, no requirement selection, no
    classification, no skill verification, no evidence inspection, and no
    re-interpretation of previous node outputs happens here.
    """
    missing = [key for key in agent_module.REQUIRED_EVIDENCE if key not in state]
    if missing:
        raise service_module.AdvisoryFitProviderError(
            "Advisory Fit agent finished without a complete analysis"
        )
    request = state["request"]
    verification = state["skill_verification"]
    levels = state["level_evaluations"]
    classifications = state["requirement_classifications"]

    assessments: list[Assessment] = []
    for req in request.project_requirements:
        key = req.skill.casefold()
        entry = verification.get(key)
        approach_relevance = classifications[key]
        assessments.append(
            Assessment(
                skill=req.skill,
                required_level=req.level,
                contributor_level=(
                    entry["contributor_level"] if entry else None
                ),
                skill_match=(
                    entry["skill_match"] if entry else "NOT_EVIDENCED"
                ),
                level_match=levels.get(key, "MISSING"),
                approach_relevance=approach_relevance,
                explanation=(
                    entry["explanation"] if entry else _NOT_SELECTED_EXPLANATION
                ),
            )
        )

    fit_percentage = calculate_fit_percentage(assessments)
    summary = stages._build_summary(assessments, fit_percentage)
    return {
        "requirement_assessments": assessments,
        "fit_percentage": fit_percentage,
        "summary": summary,
        "final_result": AdvisoryFitResult(
            fit_percentage=fit_percentage,
            assessments=assessments,
            summary=summary,
        ),
    }


def build_workflow_graph() -> StateGraph:
    """Build the Advisory Fit workflow.

    Topology::

        START
        → initialize_state
        → understand_approach
        → select_relevant_requirements
        → match_skills_and_evidence
        → calculate_fit
        → END
    """
    builder = StateGraph(AgentState)
    builder.add_node("initialize_state", agent_module.initialize_state)
    builder.add_node("understand_approach", understand_approach)
    builder.add_node(
        "select_relevant_requirements", select_relevant_requirements
    )
    builder.add_node(
        "match_skills_and_evidence", match_skills_and_evidence
    )
    builder.add_node("calculate_fit", calculate_fit)

    builder.add_edge(START, "initialize_state")
    builder.add_edge("initialize_state", "understand_approach")
    builder.add_edge(
        "understand_approach", "select_relevant_requirements"
    )
    builder.add_edge(
        "select_relevant_requirements", "match_skills_and_evidence"
    )
    builder.add_edge("match_skills_and_evidence", "calculate_fit")
    builder.add_edge("calculate_fit", END)
    return builder
