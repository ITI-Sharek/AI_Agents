from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from sharek_agents.agents.advisory_fit import agent as agent_module
from sharek_agents.agents.advisory_fit import service as service_module
from sharek_agents.agents.advisory_fit import stages
from sharek_agents.agents.advisory_fit.llm import get_advisory_fit_llm
from sharek_agents.agents.advisory_fit.prompts import (
    APPROACH_ANALYSIS_HUMAN_PROMPT,
    APPROACH_ANALYSIS_SYSTEM_PROMPT,
    RELATION_CLASSIFICATION_HUMAN_PROMPT,
    RELATION_CLASSIFICATION_SYSTEM_PROMPT,
    UNDERSTAND_APPROACH_HUMAN_PROMPT,
    UNDERSTAND_APPROACH_SYSTEM_PROMPT,
    apply_output_language,
)
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
    ApproachAnalysis,
    Assessment,
    Evidence,
    EvidenceUnderstanding,
    LevelMatch,
    RequirementRelationAnalysis,
    SkillItem,
)
from sharek_agents.agents.advisory_fit.scoring import (
    calculate_fit_percentage,
    calculate_level_match,
)
from sharek_agents.agents.advisory_fit.state import AgentState, SkillVerificationEntry
from sharek_agents.config import settings

_MAX_UNDERSTANDING_ITEMS = 50
_MAX_APPROACH_ITEMS = 50

_StructuredModel = TypeVar("_StructuredModel", bound=BaseModel)


async def _structured_extraction_llm(
    system_prompt: str,
    human_prompt: str,
    schema: type[_StructuredModel],
    input_data: AdvisoryFitInput,
    extra_vars: dict | None = None,
) -> _StructuredModel:
    """Run one bounded structured LLM extraction owned by a workflow node.

    The project context (title/description), the Approach (the primary
    subject), and the Contributor Evidence (supporting context) are always
    supplied: the Approach and Evidence are serialized as opaque, untrusted
    data and the prompt treats them as data only. The Approach is a free-text
    work description / requested contribution / technical intent that may
    come from the Project Owner, the Contributor, or another authorized
    source — never assumed to be the contributor's own plan. ``extra_vars``
    are merged into the prompt variables for calls that need additional
    bounded context (e.g. relation classification). The requested output
    language (``input_data.answer``) is appended to the system prompt as a
    high-priority instruction: it controls only the language of generated
    natural-language text and never identifiers, enum values, levels, or
    classifications (an empty ``answer`` appends nothing). The LLM is
    resolved via the dedicated Advisory Fit factory
    (``advisory_fit/llm.py``), which uses the dedicated ``ADVISORY_FIT_LLM_*``
    settings when configured and falls back to the shared configuration
    otherwise. The structured output is validated against ``schema`` and the
    call is bounded by the configured timeout.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", apply_output_language(system_prompt, input_data.answer)),
            ("human", human_prompt),
        ]
    )

    variables: dict = {
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
    if extra_vars:
        variables.update(extra_vars)

    structured = get_advisory_fit_llm().with_structured_output(
        schema, method="function_calling"
    )
    result = await asyncio.wait_for(
        (prompt | structured).ainvoke(variables),
        timeout=settings.ai_skill_profile_timeout_seconds,
    )
    return schema.model_validate(result)


async def _invoke_understand_approach_llm(
    input_data: AdvisoryFitInput,
) -> EvidenceUnderstanding:
    """Evidence-grounded understanding of the Approach.

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
    """Understanding of the work described by the Approach + requirement relevance.

    ``_structured_extraction_llm`` with the ``APPROACH_ANALYSIS_*`` prompt pair
    and the ``ApproachAnalysis`` schema. The extraction describes ONLY the
    described work and the semantic relevance of each authoritative project
    requirement to it. The authoritative requirement skill NAMES (never
    levels) are supplied as ``extra_vars``; the LLM classifies relevance
    (``DIRECT`` / ``RELATED`` / ``NOT_RELEVANT``) but never invents, renames,
    or rewords requirements, and never inspects contributor skills or levels.
    Python validates every referenced requirement against the authoritative
    request data and fails closed.
    """
    return await _structured_extraction_llm(
        APPROACH_ANALYSIS_SYSTEM_PROMPT,
        APPROACH_ANALYSIS_HUMAN_PROMPT,
        ApproachAnalysis,
        input_data,
        extra_vars={
            "requirement_skills": json.dumps(
                [r.skill for r in input_data.project_requirements], indent=2
            ),
        },
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


def _validate_approach_analysis(
    analysis: ApproachAnalysis,
    project_requirements: list[SkillItem],
) -> None:
    """Deterministic anti-hallucination checks on the approach extraction.

    Fail-closed: per-category item counts are capped; item uniqueness,
    non-emptiness, and item length are enforced by the schema. Every
    requirement relation must reference an authoritative project requirement
    exactly once (no invented requirements, no duplicates, no renames — the
    requirement reference is casefolded against the authoritative list, and
    at most one record per requirement), and the record count is bounded by
    the number of authoritative requirements. Requirements not covered by a
    record are NOT rejected: they normalize to ``NOT_RELEVANT`` downstream
    (the safe result when the LLM is silent). Nothing here inspects
    contributor skills, computes scores, or performs matching.
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

    requirement_keys = [r.skill.casefold() for r in project_requirements]
    max_records = len(requirement_keys)
    if len(analysis.requirement_relations) > max_records:
        raise service_module.AdvisoryFitProviderError(
            "approach analysis contains too many requirement relations: "
            f"{len(analysis.requirement_relations)} (max {max_records})"
        )
    seen: set[str] = set()
    for relation in analysis.requirement_relations:
        key = relation.requirement.casefold()
        if key not in requirement_keys:
            raise service_module.AdvisoryFitProviderError(
                "approach analysis references nonexistent requirement: "
                f"'{relation.requirement}'"
            )
        if key in seen:
            raise service_module.AdvisoryFitProviderError(
                f"duplicate approach relation for requirement: "
                f"'{relation.requirement}'"
            )
        seen.add(key)


async def understand_approach(state: AgentState) -> dict:
    """Understand the Approach — the primary workflow artifact.

    Reads: ``request``.
    Writes: ``approach_analysis``, ``intended_approach``.

    The node is responsible ONLY for understanding the Approach — a free-text
    work description / requested contribution / technical intent that may be
    written by the Project Owner, the Contributor, or another authorized
    source. Two bounded structured LLM calls run here:

    - ``EvidenceUnderstanding`` (``approach_analysis``): what the contributor
      has built, grounded in the Evidence, with the Approach as context. This
      extraction is requirement-agnostic.
    - ``ApproachAnalysis`` (``intended_approach``): the work described by the
      Approach, from the Approach as the primary subject and the Evidence
      strictly as supporting context, PLUS the semantic relevance of each
      authoritative project requirement to the described work. The LLM is the
      single semantic authority for that relevance (``DIRECT`` / ``RELATED`` /
      ``NOT_RELEVANT``); it never invents requirements and never sees or emits
      levels or scores.

    Both extractions are validated deterministically (evidence reference
    bounds, uniqueness, caps; requirement-relation references, duplicates,
    and caps) and never inspect contributor skills, generate a roadmap, or
    perform requirement matching. An empty Approach yields an empty
    ``ApproachAnalysis`` without an LLM call.
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
        _validate_approach_analysis(analysis, request.project_requirements)
    else:
        analysis = ApproachAnalysis()

    return {"approach_analysis": understanding, "intended_approach": analysis}


# Mapping from the LLM's semantic approach-relevance verdicts to the
# response representation consumed by ``calculate_fit`` (the
# ``ApproachRelevance`` literal). ``DIRECT`` maps to full approach credit,
# ``RELATED`` to partial credit (the described work is meaningfully related
# but does not directly target the requirement), and ``NOT_RELEVANT`` to no
# credit. Python owns this mapping; the LLM never emits a score.
_APPROACH_RELEVANCE_MAP: dict[str, str] = {
    "DIRECT": "DIRECT",
    "RELATED": "PARTIAL",
    "NOT_RELEVANT": "NOT_MENTIONED",
}


def select_relevant_requirements(state: AgentState) -> dict:
    """Select project requirements from the LLM-classified approach relations.

    Reads: ``request``, ``intended_approach``.
    Writes: ``relevant_requirements``, ``partially_relevant_requirements``,
    ``requirement_classifications``.

    Deterministic selection driven ONLY by the ``requirement_relations`` of
    the ``ApproachAnalysis`` written as ``intended_approach`` by
    ``understand_approach`` — the LLM-classified semantic relevance of each
    authoritative project requirement to the work described by the Approach
    (whether the Approach was written by the Project Owner, the Contributor,
    or another authorized source). The LLM is the single semantic authority
    for that relevance; this node never re-derives, re-lexes, or re-classifies
    it and no lexical/string matching happens anywhere in the path.
    Contributor skills, contributor evidence, project context, and any
    matching/fit results are never inspected; no skill verification, fit
    calculation, or scoring happens here, and no LLM call is made.

    Every project requirement is classified exactly once: ``DIRECT`` or
    ``RELATED`` verdicts select the requirement for matching (in request
    order), ``NOT_RELEVANT`` excludes it, and a requirement the LLM omitted
    (silence never implies relevance) also normalizes to ``NOT_RELEVANT``.
    This node is the ONLY producer of the per-requirement relevance mapping:
    it writes, for EVERY project requirement, an explicit
    ``requirement_classifications`` entry (normalized requirement name to the
    response ``DIRECT`` / ``PARTIAL`` / ``NOT_MENTIONED`` value via
    ``_APPROACH_RELEVANCE_MAP``) so downstream nodes consume the decision
    verbatim instead of reconstructing it. The ordered partitions
    ``relevant_requirements`` (DIRECT or RELATED, in request order) and
    ``partially_relevant_requirements`` (the RELATED subset) are kept for the
    matching node, which consumes them unchanged.
    """
    if "intended_approach" not in state:
        raise service_module.AdvisoryFitProviderError(
            "select_relevant_requirements requires understand_approach to "
            "have produced an analysis first"
        )
    request = state["request"]
    analysis = state["intended_approach"]

    verdicts = {
        relation.requirement.casefold(): relation.relevance
        for relation in analysis.requirement_relations
    }

    relevant_requirements: list[str] = []
    partially_relevant_requirements: list[str] = []
    requirement_classifications: dict[str, str] = {}

    for requirement in request.project_requirements:
        key = requirement.skill.casefold()
        verdict = verdicts.get(key, "NOT_RELEVANT")
        requirement_classifications[key] = _APPROACH_RELEVANCE_MAP[verdict]
        if verdict == "RELATED":
            partially_relevant_requirements.append(requirement.skill)
        if verdict in ("DIRECT", "RELATED"):
            relevant_requirements.append(requirement.skill)

    return {
        "relevant_requirements": relevant_requirements,
        "partially_relevant_requirements": partially_relevant_requirements,
        "requirement_classifications": requirement_classifications,
    }


def _declared_skill_map(skills) -> dict[str, str]:
    """Map casefolded skill -> declared level from ``contributor.skills``."""
    return {s.skill.casefold(): s.level for s in skills}


def _validate_relation_analysis(
    analysis: RequirementRelationAnalysis,
    relevant: list[str],
    declared_skill_names: list[str],
    evidence_count: int,
) -> None:
    """Deterministic fail-closed validation of the relation LLM output.

    Rejects: too many records, nonexistent requirement references, duplicate
    requirement references, nonexistent declared-skill references in
    ``related_skills``, and out-of-range evidence indexes. Requirements not
    covered by a record are NOT rejected: they default to ``MISSING``
    (the safe result when the LLM is uncertain or silent).
    """
    relevant_keys = [skill.casefold() for skill in relevant]
    declared_keys = set(declared_skill_names)
    max_records = len(relevant)

    def check_requirement_records(records, label: str) -> None:
        if len(records) > max_records:
            raise service_module.AdvisoryFitProviderError(
                f"relation analysis contains too many {label}: "
                f"{len(records)} (max {max_records})"
            )
        seen: set[str] = set()
        for record in records:
            key = record.requirement_skill.casefold()
            if key not in relevant_keys:
                raise service_module.AdvisoryFitProviderError(
                    f"{label} references nonexistent requirement: "
                    f"'{record.requirement_skill}'"
                )
            if key in seen:
                raise service_module.AdvisoryFitProviderError(
                    f"duplicate {label} for requirement: "
                    f"'{record.requirement_skill}'"
                )
            seen.add(key)

    check_requirement_records(analysis.skill_relations, "skill relation")
    check_requirement_records(
        analysis.evidence_relations, "evidence relation"
    )

    for record in analysis.skill_relations:
        for skill in record.related_skills:
            if skill.casefold() not in declared_keys:
                raise service_module.AdvisoryFitProviderError(
                    "skill relation references nonexistent declared skill: "
                    f"'{skill}'"
                )

    for record in analysis.evidence_relations:
        out_of_range = [
            i for i in record.evidence_indexes if i < 0 or i >= evidence_count
        ]
        if out_of_range:
            raise service_module.AdvisoryFitProviderError(
                "evidence relation references out-of-range evidence index: "
                f"{sorted(out_of_range)}"
            )


async def _invoke_relation_classification_llm(
    input_data: AdvisoryFitInput,
    relevant: list[str],
    understanding: EvidenceUnderstanding,
) -> RequirementRelationAnalysis:
    """Classify requirement ↔ skill and requirement ↔ evidence relations.

    ``_structured_extraction_llm`` with the ``RELATION_CLASSIFICATION_*``
    prompt pair and the ``RequirementRelationAnalysis`` schema. The LLM is
    given the relevant requirement skill names, the contributor's complete
    declared skill NAMES (never levels), the raw evidence records, and the
    structured ``EvidenceUnderstanding`` (reusing the existing evidence
    extraction instead of duplicating it). It returns exactly one
    ``MATCHED`` / ``RELATED`` / ``MISSING`` relation per requirement per
    section; Python validates and defaults uncovered requirements to
    ``MISSING``.
    """
    return await _structured_extraction_llm(
        RELATION_CLASSIFICATION_SYSTEM_PROMPT,
        RELATION_CLASSIFICATION_HUMAN_PROMPT,
        RequirementRelationAnalysis,
        input_data,
        extra_vars={
            "requirement_skills": json.dumps(relevant, indent=2),
            "contributor_skills": json.dumps(
                [s.skill for s in input_data.contributor.skills], indent=2
            ),
            "evidence_understanding": json.dumps(
                understanding.model_dump(mode="json", by_alias=True),
                indent=2,
            ),
        },
    )


_NO_RELATION_EXPLANATION = (
    "No declared contributor skills or evidence to relate; relation defaults "
    "to MISSING."
)


def _relation_explanation(
    skill_relation, evidence_relation
) -> str:
    """Combine the LLM's per-requirement relation explanations."""
    parts: list[str] = []
    if skill_relation:
        parts.append(f"Skill relation: {skill_relation.explanation}")
    if evidence_relation:
        parts.append(f"Evidence relation: {evidence_relation.explanation}")
    if not parts:
        return (
            "Requirement relation not reported by the relation analysis; "
            "defaulted to MISSING."
        )
    return " ".join(parts)


async def match_skills_and_evidence(state: AgentState) -> dict:
    """Verify the contributor can execute the selected requirements only.

    Reads: ``relevant_requirements``, ``approach_analysis``, ``request``
    (``contributor.skills``, ``contributor.evidence``, and the required
    levels of ``project_requirements``).
    Writes: ``skill_verification``, ``level_evaluations``.

    The node is responsible ONLY for (a) the bounded LLM semantic relation
    classification of the selection produced by
    ``select_relevant_requirements`` and (b) deterministic level verification
    of that selection. For every relevant requirement it:

    1. Classifies the requirement ↔ declared-skill semantic relation
       (``MATCHED`` / ``RELATED`` / ``MISSING``) via the relation LLM call —
       the single authority for the relation; Python never re-derives it from
       lexical matching.
    2. Classifies the requirement ↔ evidence semantic relation (``MATCHED`` /
       ``RELATED`` / ``MISSING``) via the same call, reusing the structured
       ``EvidenceUnderstanding`` (``approach_analysis``) as input.
    3. Deterministically compares the declared level against the required
       level. The declared level is always taken from the authoritative
       ``contributor.skills`` entry whose normalized name equals the
       requirement name; a skill declared under a different name never
       supplies a level — so a ``RELATED`` classification never creates or
       implies a level match.

    The LLM output is validated fail-closed (requirement references,
    declared-skill references, evidence indexes, duplicates, caps);
    requirements not covered by a returned record default to ``MISSING`` —
    the safe result — never to ``MATCHED``.

    Skill, evidence, and level are three independent dimensions and are never
    conflated: a skill can be ``MATCHED`` with ``MISSING`` evidence,
    ``RELATED`` with ``MATCHED`` evidence, or ``MATCHED`` with an
    insufficient level. The declared skill level remains the authoritative
    level for ``calculate_level_match``; evidence support never replaces or
    rewrites a declared level, and an undeclared skill is never fabricated
    into a declaration.

    Requirement selection and fit calculation are out of scope: the node
    never determines relevance, never interprets the Approach, never reads
    the project description or title, and never computes a percentage. The
    output holds only the matching data ``calculate_fit`` consumes: no
    relevance values, no fit scores, no recommendations.
    """
    if "relevant_requirements" not in state:
        raise service_module.AdvisoryFitProviderError(
            "match_skills_and_evidence requires select_relevant_requirements "
            "to have produced a selection first"
        )
    if "approach_analysis" not in state:
        raise service_module.AdvisoryFitProviderError(
            "match_skills_and_evidence requires understand_approach to have "
            "produced an evidence understanding first"
        )
    request = state["request"]
    relevant = state["relevant_requirements"]
    understanding = state["approach_analysis"]

    declared_levels = _declared_skill_map(request.contributor.skills)
    required_levels = {
        req.skill.casefold(): req.level for req in request.project_requirements
    }

    analysis = RequirementRelationAnalysis()
    relation_run = bool(
        relevant and (declared_levels or request.contributor.evidence)
    )
    if relation_run:
        try:
            analysis = await _invoke_relation_classification_llm(
                request, relevant, understanding
            )
        except ValidationError as exc:
            raise service_module.AdvisoryFitProviderError(
                f"Advisory Fit provider returned invalid output: {exc}"
            ) from exc
        _validate_relation_analysis(
            analysis,
            relevant,
            list(declared_levels),
            len(request.contributor.evidence),
        )

    skill_relations = {
        record.requirement_skill.casefold(): record
        for record in analysis.skill_relations
    }
    evidence_relations = {
        record.requirement_skill.casefold(): record
        for record in analysis.evidence_relations
    }

    skill_verification: dict[str, SkillVerificationEntry] = {}
    level_evaluations: dict[str, LevelMatch] = {}

    for skill in relevant:
        key = skill.casefold()
        required_level = required_levels[key]
        declared_level = declared_levels.get(key)
        skill_relation = skill_relations.get(key)
        evidence_relation = evidence_relations.get(key)

        skill_verification[key] = SkillVerificationEntry(
            contributor_level=declared_level,
            skill_match=(
                skill_relation.relation if skill_relation else "MISSING"
            ),
            evidence_match=(
                evidence_relation.relation if evidence_relation else "MISSING"
            ),
            explanation=(
                _relation_explanation(skill_relation, evidence_relation)
                if relation_run
                else _NO_RELATION_EXPLANATION
            ),
        )
        level_evaluations[key] = calculate_level_match(
            required_level, declared_level
        )

    return {
        "skill_verification": skill_verification,
        "level_evaluations": level_evaluations,
    }


_NOT_SELECTED_EXPLANATION = (
    "Requirement not selected: no skills, evidence, or level matching was "
    "performed."
)


def calculate_fit(state: AgentState) -> dict:
    """Generate the final Advisory Fit — the pure aggregation stage.

    Reads: ``requirement_classifications``, ``skill_verification``,
    ``level_evaluations``, and ``request`` (``project_requirements`` only,
    for the per-requirement enumeration and required levels).
    Writes: ``requirement_assessments``, ``fit_percentage``, ``summary``,
    ``final_result`` (including the descriptive ``matched_skills`` /
    ``evaluated_skills`` summary counts).

    This node is the ONLY node responsible for producing the final Advisory
    Fit result and is a pure aggregation stage: it combines the previous
    nodes' deterministic outputs into one ``Assessment`` per project
    requirement, in request order. The schema-required ``approach_relevance``
    is read verbatim from the ``requirement_classifications`` mapping written
    by ``select_relevant_requirements`` — that node is the only producer of
    the relevance classification, and this node never infers, reconstructs,
    classifies, or decides relevance. The matching data
    (``skill_verification``, ``level_evaluations`` — including the
    deterministic ``evidence_match`` classification) is attached for every
    requirement; requirements with no matching data are reported with
    ``MISSING``/``MISSING`` and a fixed explanation. The fit percentage
    reuses the existing ``calculate_fit_percentage`` scoring (now including
    the evidence weight), and the public result reuses the existing
    ``AdvisoryFitResult`` response schema. The descriptive
    ``evaluated_skills`` (requirements actually evaluated — DIRECT or RELATED
    approach relevance, never NOT_MENTIONED) and ``matched_skills`` (evaluated
    requirements whose ``skill_match`` is ``MATCHED``) counts are derived
    directly from the assessments as metadata and never feed the percentage.
    No LLM reasoning, no requirement selection, no classification, no skill
    verification, no evidence inspection, and no re-interpretation of
    previous node outputs happens here.
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
                    entry["skill_match"] if entry else "MISSING"
                ),
                level_match=levels.get(key, "MISSING"),
                evidence_match=(
                    entry["evidence_match"] if entry else "MISSING"
                ),
                approach_relevance=approach_relevance,
                explanation=(
                    entry["explanation"] if entry else _NOT_SELECTED_EXPLANATION
                ),
            )
        )

    fit_percentage = calculate_fit_percentage(assessments)
    summary = stages._build_summary(assessments, fit_percentage)
    evaluated = [
        a for a in assessments if a.approach_relevance != "NOT_MENTIONED"
    ]
    evaluated_skills = len(evaluated)
    matched_skills = sum(1 for a in evaluated if a.skill_match == "MATCHED")
    return {
        "requirement_assessments": assessments,
        "fit_percentage": fit_percentage,
        "summary": summary,
        "final_result": AdvisoryFitResult(
            fit_percentage=fit_percentage,
            matched_skills=matched_skills,
            evaluated_skills=evaluated_skills,
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
