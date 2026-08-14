"""Deterministic skill-profile construction and validation (Phase 15).

Turns the agent's final answer text into a structured skill profile
that follows the public response contract:

* ``level`` is exactly ``beginner | intermediate | advanced | expert``
  (the legacy LLM key ``proficiency`` is accepted as an internal alias
  and mapped explicitly to the public ``level``),
* every kept skill exposes a human-readable ``evidence`` string
  describing the supporting evidence (the legacy ``evidence_summary``
  key is mapped to it when present),
* every kept skill cites at least one real evidence ID collected during
  this agent run,
* when the run collected framework-detection evidence, the skill names
  are limited to the detected technologies (``detections[].name`` of
  the stored detection reports) — the detected technologies are the
  authoritative skill candidates the LLM is asked to evaluate, and the
  LLM is never allowed to emit invented skill names. Detection only
  provides the candidate names and proves presence; proficiency is a
  prompt-level concern inferred by the LLM from the other evidence,
  never by this module,
* detection is REQUIRED: when the run collected NO framework-detection
  evidence at all, the skill-name allowlist is empty rather than
  unrestricted, so no invented name can survive and the result is an
  empty insufficient-evidence profile,
* every accepted skill name is canonicalized to the detected technology
  display name it matched (matching is case-insensitive; the emitted
  casing is never kept),
* analysis-mode citation scoping is enforced from the collected
  ``EvidenceRecord.scope`` metadata: CONTRIBUTOR claims must cite
  contributor-scoped evidence when such evidence exists; PROJECT
  analysis never uses contributor-scoped evidence,
* duplicate evidence IDs are deduplicated,
* invalid level/confidence/name entries are rejected,
* skills without a non-empty evidence description are rejected — the
  public contract requires it,
* skills whose citations cannot be validated are rejected — the LLM is
  never trusted to validate its own citations,
* the contract maximum of ``MAX_SKILLS = 20`` is enforced by
  deterministic truncation with a recorded note,
* unsupported profiles are returned as an insufficient-evidence result
  instead of fabricated claims.

All validation is deterministic and operates on the run's
``EvidenceBundle`` only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from sharek_agents.agents.skill_profiling_agent.evidence import EvidenceBundle
from sharek_agents.agents.skill_profiling_agent.evidence_context import (
    ANALYSIS_MODE_CONTRIBUTOR,
    ANALYSIS_MODE_PROJECT,
)
from sharek_agents.agents.skill_profiling_agent.schemas import (
    AgentSkill,
    SkillProfileAgentOutput,
)

logger = logging.getLogger(__name__)

MAX_SKILLS = 20
MAX_NOTES = 20

# Evidence type of the deterministic framework-detection tool results
# (see ``evidence._EVIDENCE_TYPE_BY_TOOL``: ``detect_frameworks`` is
# recorded under this type with its full ``DetectionReport`` JSON as
# the record result).
_DETECTION_EVIDENCE_TYPE = "framework_detection"

# Public level scale. Never invent another one.
LEVELS = ("beginner", "intermediate", "advanced", "expert")

# Backward-compatible alias for the old package export.
PROFICIENCY_LEVELS = LEVELS


class AgentSkillCandidate(BaseModel):
    """Lenient parse model for one LLM-emitted skill.

    ``evidence_ids`` may be empty here; the builder repairs/rejects
    citations afterwards. Level, confidence, and name are strict. The
    legacy ``proficiency`` and ``evidence_summary`` keys are accepted as
    internal aliases and mapped to the public ``level`` / ``evidence``.
    """

    name: str = Field(min_length=1, max_length=100)
    level: Literal["beginner", "intermediate", "advanced", "expert"] | None = None
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"] | None = (
        None
    )
    confidence: float = Field(ge=0, le=1)
    evidence: str | None = None
    evidence_summary: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def build_skill_profile(
    answer: str,
    evidence: EvidenceBundle,
    *,
    analysis_mode: str | None = None,
) -> SkillProfileAgentOutput:
    """Deterministically build the validated skill profile from the answer.

    ``answer`` is the agent's final text; it must be a JSON object
    containing a ``skills`` list. Anything else yields an
    insufficient-evidence result. Skills that fail structural validation,
    carry no evidence description, or cite no real collected evidence are
    rejected with a recorded note.

    ``analysis_mode`` ("CONTRIBUTOR" or "PROJECT") selects the
    deterministic citation-scoping rules (Contributor Profile). It is a
    request property that the ``EvidenceBundle`` does not store, so the
    caller passes it explicitly; ``None`` disables mode scoping.
    """
    notes: list[str] = []

    profile_dict = _try_parse_profile(answer)
    if profile_dict is None:
        return SkillProfileAgentOutput(
            skills=[],
            insufficient_evidence=True,
            recommendation="needs_more_evidence",
            message=(
                "The agent's final answer was not a structured skill "
                "profile (expected a JSON object with a 'skills' list)."
            ),
        )

    raw_skills = profile_dict.get("skills")
    if not isinstance(raw_skills, list):
        return SkillProfileAgentOutput(
            skills=[],
            insufficient_evidence=True,
            recommendation="needs_more_evidence",
            message=("The agent's final answer did not contain a 'skills' list."),
        )

    skills: list[AgentSkill] = []
    detected_names: set[str] | None = _detected_technology_names(evidence)
    detection_evidence_missing = detected_names is None
    detected_names = detected_names or set()
    allowed_name_folds: frozenset[str] = frozenset(
        name.casefold() for name in detected_names
    )
    canonical_name_by_fold: dict[str, str] = {
        name.casefold(): name for name in detected_names
    }
    contributor_scoped_ids: frozenset[str] = frozenset(
        record.evidence_id
        for record in evidence.records()
        if record.scope == "contributor"
    )
    mode = (analysis_mode or "").strip()

    for index, item in enumerate(raw_skills):
        if not isinstance(item, dict):
            notes.append(f"skill #{index + 1} rejected: not a JSON object")
            continue
        try:
            candidate = AgentSkillCandidate.model_validate(item)
        except ValidationError:
            notes.append(
                f"skill #{index + 1} rejected: invalid name, level "
                "(must be beginner, intermediate, advanced, or expert), "
                "or confidence (must be between 0 and 1)"
            )
            continue

        level = candidate.level or candidate.proficiency
        if level is None:
            notes.append(
                f"skill '{candidate.name}' rejected: no level "
                "(must be beginner, intermediate, advanced, or expert)"
            )
            continue

        evidence_text = (candidate.evidence or candidate.evidence_summary or "").strip()
        if not evidence_text:
            notes.append(
                f"skill '{candidate.name}' rejected: no evidence description "
                "(the public contract requires a human-readable 'evidence' "
                "string)"
            )
            continue

        if candidate.name.casefold() not in allowed_name_folds:
            if detection_evidence_missing:
                notes.append(
                    f"skill '{candidate.name}' rejected: no framework "
                    "detection evidence was collected (the detected "
                    "technologies are the authoritative skill candidates)"
                )
            else:
                notes.append(
                    f"skill '{candidate.name}' rejected: not a detected "
                    "technology (the detected technologies are the "
                    "authoritative skill candidates)"
                )
            continue

        valid_ids = _validated_evidence_ids(candidate.evidence_ids, evidence)
        if not valid_ids:
            notes.append(
                f"skill '{candidate.name}' rejected: no valid evidence "
                "citations (citations must reference evidence collected "
                "during this run)"
            )
            continue

        if (
            mode == ANALYSIS_MODE_CONTRIBUTOR
            and contributor_scoped_ids
            and not (set(valid_ids) & contributor_scoped_ids)
        ):
            notes.append(
                f"skill '{candidate.name}' rejected: contributor-specific "
                "claims must cite contributor-scoped evidence when such "
                "evidence exists (cited only repository-wide or request "
                "evidence)"
            )
            continue

        if mode == ANALYSIS_MODE_PROJECT and (set(valid_ids) & contributor_scoped_ids):
            notes.append(
                f"skill '{candidate.name}' rejected: contributor-scoped "
                "evidence must not be used in PROJECT analysis (the "
                "profile is repository-wide)"
            )
            continue

        skills.append(
            AgentSkill(
                name=canonical_name_by_fold[candidate.name.casefold()],
                level=level,
                confidence=candidate.confidence,
                evidence=evidence_text,
                evidence_ids=valid_ids,
                limitations=_clean_limitations(candidate.limitations),
            )
        )

    if len(skills) > MAX_SKILLS:
        dropped = len(skills) - MAX_SKILLS
        skills = skills[:MAX_SKILLS]
        notes.append(f"{dropped} skill(s) dropped: exceeds MAX_SKILLS={MAX_SKILLS}")

    if (
        mode == ANALYSIS_MODE_CONTRIBUTOR
        and not contributor_scoped_ids
        and skills
    ):
        notes.append(
            "no contributor-scoped evidence was collected: contributor "
            "authorship of the emitted skills could not be "
            "deterministically verified"
        )

    insufficient = not skills
    recommendation: Literal["pending_review", "needs_more_evidence"] = (
        "needs_more_evidence" if insufficient else "pending_review"
    )
    if not notes:
        message = (
            "No skills could be derived from the collected evidence."
            if insufficient
            else "Skill profile derived from validated evidence."
        )
    else:
        message = "; ".join(notes[:MAX_NOTES])
        if len(notes) > MAX_NOTES:
            message += f"; ... {len(notes) - MAX_NOTES} more"

    return SkillProfileAgentOutput(
        skills=skills,
        insufficient_evidence=insufficient,
        recommendation=recommendation,
        message=message,
    )


def _detected_technology_names(evidence: EvidenceBundle) -> set[str] | None:
    """The detected technologies of this run, or ``None`` when none exist.

    Reads only the run's collected evidence: every stored
    framework-detection record carries the ``DetectionReport`` JSON in
    its ``result``, whose ``detections[].name`` entries are the
    deterministic technology display names. Returns ``None`` when the
    run collected NO detection evidence at all — the caller then treats
    the skill-name allowlist as empty (detection is required, so no
    skill name is allowed); returns the possibly-empty set when
    detection evidence exists (an empty set allows no skill names).
    """
    detected: set[str] = set()
    found_detection_evidence = False
    for record in evidence.records():
        if record.evidence_type != _DETECTION_EVIDENCE_TYPE:
            continue
        found_detection_evidence = True
        try:
            payload = json.loads(record.result)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        detections = payload.get("detections")
        if not isinstance(detections, list):
            continue
        for item in detections:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    detected.add(name.strip())
    if not found_detection_evidence:
        return None
    return detected


def _validated_evidence_ids(
    cited: list[str],
    evidence: EvidenceBundle,
) -> list[str]:
    """Keep only citations that reference evidence collected this run.

    Order is preserved; duplicates are deduplicated deterministically.
    """
    return list(
        dict.fromkeys(
            evidence_id
            for evidence_id in (cited or [])
            if isinstance(evidence_id, str) and evidence.contains(evidence_id)
        )
    )


def _clean_limitations(limitations: list[str] | None) -> list[str]:
    """Keep non-empty, deduplicated limitations in the given order."""
    return list(
        dict.fromkeys(
            item.strip()
            for item in (limitations or [])
            if isinstance(item, str) and item.strip()
        )
    )


def _try_parse_profile(answer: str) -> dict[str, Any] | None:
    """Parse the final answer as a JSON object, tolerating code fences."""
    if not answer or not answer.strip():
        return None

    candidates = [answer]
    stripped = answer.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        if lines:
            candidates.append("\n".join(lines))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
