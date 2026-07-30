"""Post-Agent evidence grounding and output validation layer.

Validates the final structured result produced by the ReAct Agent
before it is returned by the Documentation Understanding API.

The validator is deterministic, makes no LLM calls, and operates
exclusively on the current request-scoped context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from sharek_agents.agents.document_understanding.memory_store import InMemoryStore
from sharek_agents.agents.document_understanding.schemas import (
    CloudinaryResourceRef,
    Conflict,
    DocumentUnderstandingResult,
    EvidenceItem,
)


# ── Text normalisation ─────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Normalize text for safe comparison, ignoring whitespace differences."""
    return " ".join(text.split())


def _ref_key(ref: CloudinaryResourceRef | None) -> str | None:
    """Derive a simple string identifier from a CloudinaryResourceRef."""
    if ref is None:
        return None
    return ref.public_id or ref.url or None


def _refs_match(
    a: CloudinaryResourceRef | None,
    b: CloudinaryResourceRef | None,
) -> bool:
    """Compare two CloudinaryResourceRef values for logical equality.

    Comparison is based on the identifying fields (public_id, url).
    Non-identifying fields (resource_type, delivery_type, etc.) are
    ignored because they may differ between the agent output and the
    original source metadata while still referring to the same document.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_key = a.public_id or a.url
    b_key = b.public_id or b.url
    if a_key and b_key:
        return a_key == b_key
    return a.model_dump(exclude_none=True) == b.model_dump(exclude_none=True)


# ── README detection ───────────────────────────────────────────────────────────


_README_PATTERNS = [
    re.compile(r"^#\s+\S", re.MULTILINE),
    re.compile(r"^##\s+\S", re.MULTILINE),
    re.compile(r"\bREADME\b"),
    re.compile(r"^#\s*(Overview|Introduction|Getting Started|Installation)", re.MULTILINE),
]


def _contains_readme_content(text: str) -> bool:
    """Return True if *text* contains README-like patterns."""
    return any(p.search(text) for p in _README_PATTERNS)


# ── Validation context ─────────────────────────────────────────────────────────


@dataclass
class ValidationContext:
    """Request-scoped context for post-agent validation.

    Attributes:
        vector_store: The request-scoped in-memory vector store containing
            all indexed document chunks from the current request.
        document_ref_keys: Optional set of valid document reference keys
            for document-level validation when ``chunk_id`` is unavailable.
    """
    vector_store: InMemoryStore
    document_ref_keys: set[str] = field(default_factory=set)


# ── Validation result models ───────────────────────────────────────────────────


class EvidenceValidationEntry(BaseModel):
    """A single evidence item with a validation issue."""
    claim: str = Field(description="The evidence claim")
    chunk_id: str | None = Field(default=None, description="Referenced chunk ID")
    issue: str = Field(description="Description of the validation issue")


class UnsupportedClaimEntry(BaseModel):
    """A claim that has no supporting evidence."""
    claim: str = Field(description="The unsupported claim")
    reason: str = Field(description="Why it is unsupported")


class MissingInfoInconsistency(BaseModel):
    """A missing-information entry flagged as potentially inconsistent."""
    field_path: str = Field(description="Missing info field path")
    description: str = Field(description="Missing info description")
    reason: str = Field(description="Why it appears inconsistent")


class ConflictValidationEntry(BaseModel):
    """A conflict entry with a validation issue."""
    field_path: str = Field(description="Conflict field path")
    issue: str = Field(description="Description of the validation issue")


class ValidationResult(BaseModel):
    """Structured result of post-agent validation.

    The validator never modifies the original result.  It reports
    findings so the caller can decide how to proceed.
    """
    is_valid: bool = Field(default=False, description="Overall validation outcome")
    evidence_validation_status: Literal["valid", "partial", "invalid"] = Field(
        default="valid",
        description="Aggregate status of evidence items",
    )
    invalid_evidence: list[EvidenceValidationEntry] = Field(
        default_factory=list,
        description="Evidence items with invalid or unmatched references",
    )
    unsupported_claims: list[UnsupportedClaimEntry] = Field(
        default_factory=list,
        description="Claims without any supporting evidence",
    )
    missing_information_inconsistencies: list[MissingInfoInconsistency] = Field(
        default_factory=list,
        description="Missing info entries inconsistent with available evidence",
    )
    invalid_conflicts: list[ConflictValidationEntry] = Field(
        default_factory=list,
        description="Conflicts with invalid or unsupported evidence",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Blocking validation errors",
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking validation warnings",
    )


# ── Main validator ─────────────────────────────────────────────────────────────


class DocumentUnderstandingValidator:
    """Post-agent validator for ``DocumentUnderstandingResult``.

    Validates:
    * Final output schema conformance.
    * Evidence references against the request-scoped vector store.
    * Evidence excerpts against referenced chunk content.
    * Unsupported claims without evidence grounding.
    * Missing-information consistency with available evidence.
    * Conflict validity and source grounding.
    * README exclusion from the structured output.

    The validator is deterministic and makes no external calls.
    """

    def __init__(self, context: ValidationContext, strict: bool = True) -> None:
        self._context = context
        self._strict = strict
        self._store = context.vector_store

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, result: DocumentUnderstandingResult) -> ValidationResult:
        """Validate *result* against the request-scoped context.

        Args:
            result: The agent-produced structured result.

        Returns:
            A ``ValidationResult`` with detailed findings.
        """
        output = ValidationResult()

        schema_errors = self._validate_schema(result)
        if schema_errors:
            output.validation_errors.extend(schema_errors)
            output.is_valid = False
            return output

        invalid_evidence = self._validate_evidence(result.evidence)
        output.invalid_evidence.extend(invalid_evidence)

        excerpt_issues = self._validate_excerpts(result.evidence)
        output.invalid_evidence.extend(excerpt_issues)

        unsupported = self._find_unsupported_claims(result)
        output.unsupported_claims = unsupported

        inconsistencies = self._validate_missing_information(result)
        output.missing_information_inconsistencies = inconsistencies

        invalid_conflicts = self._validate_conflicts(result.conflicts)
        output.invalid_conflicts = invalid_conflicts

        readme_warnings = self._check_readme_exclusion(result)
        output.validation_warnings.extend(readme_warnings)

        self._set_aggregate_status(output, result)

        return output

    # ── Schema validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_schema(result: DocumentUnderstandingResult) -> list[str]:
        """Validate that *result* conforms to the expected schema.

        Returns a list of error messages (empty when valid).
        """
        errors: list[str] = []

        if not result.project_id:
            errors.append("result.project_id is required")

        for field_name in (
            "project_profile", "business", "goals", "features",
            "requirements", "technical", "other_info",
        ):
            sub = getattr(result, field_name, None)
            if sub is not None:
                try:
                    sub.model_validate(sub.model_dump())
                except ValidationError as e:
                    errors.append(f"result.{field_name} is malformed: {e}")

        if not isinstance(result.evidence, list):
            errors.append("result.evidence must be a list")
        if not isinstance(result.missing_information, list):
            errors.append("result.missing_information must be a list")
        if not isinstance(result.conflicts, list):
            errors.append("result.conflicts must be a list")

        return errors

    # ── Evidence reference validation ────────────────────────────────────

    def _validate_evidence(
        self,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceValidationEntry]:
        """Validate evidence references against the vector store."""
        issues: list[EvidenceValidationEntry] = []

        for item in evidence:
            source = item.source

            if not source.chunk_id and not source.document_ref:
                issues.append(EvidenceValidationEntry(
                    claim=item.claim,
                    issue="No chunk_id or document_ref provided; evidence cannot be grounded",
                ))
                continue

            if source.chunk_id:
                chunk = self._store.get_by_chunk_id(source.chunk_id)
                if chunk is None:
                    issues.append(EvidenceValidationEntry(
                        claim=item.claim,
                        chunk_id=source.chunk_id,
                        issue=f"Referenced chunk '{source.chunk_id}' does not exist "
                              f"in the current request context",
                    ))
                    continue

                if source.document_ref is not None and chunk.document_reference is not None:
                    if not _refs_match(source.document_ref, chunk.document_reference):
                        issues.append(EvidenceValidationEntry(
                            claim=item.claim,
                            chunk_id=source.chunk_id,
                            issue="document_reference does not match the source chunk",
                        ))

                if source.filename is not None and chunk.filename is not None:
                    if source.filename != chunk.filename:
                        issues.append(EvidenceValidationEntry(
                            claim=item.claim,
                            chunk_id=source.chunk_id,
                            issue=f"filename '{source.filename}' does not match "
                                  f"chunk filename '{chunk.filename}'",
                        ))

                if source.page_number is not None and chunk.page_number is not None:
                    if source.page_number != chunk.page_number:
                        issues.append(EvidenceValidationEntry(
                            claim=item.claim,
                            chunk_id=source.chunk_id,
                            issue=f"page_number {source.page_number} does not match "
                                  f"chunk page_number {chunk.page_number}",
                        ))

                if source.section is not None and chunk.section is not None:
                    if source.section != chunk.section:
                        issues.append(EvidenceValidationEntry(
                            claim=item.claim,
                            chunk_id=source.chunk_id,
                            issue=f"section '{source.section}' does not match "
                                  f"chunk section '{chunk.section}'",
                        ))

            elif source.document_ref is not None:
                doc_key = _ref_key(source.document_ref)
                if doc_key is not None and self._context.document_ref_keys:
                    if doc_key not in self._context.document_ref_keys:
                        issues.append(EvidenceValidationEntry(
                            claim=item.claim,
                            issue=f"document_ref '{doc_key}' not found "
                                  f"in current request context",
                        ))

        return issues

    # ── Excerpt validation ───────────────────────────────────────────────

    def _validate_excerpts(
        self,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceValidationEntry]:
        """Validate that source excerpts are supported by chunk content."""
        issues: list[EvidenceValidationEntry] = []

        for item in evidence:
            source = item.source

            if not source.source_excerpt:
                continue
            if not source.chunk_id:
                issues.append(EvidenceValidationEntry(
                    claim=item.claim,
                    issue="Excerpt provided but no chunk_id to verify against",
                ))
                continue

            chunk = self._store.get_by_chunk_id(source.chunk_id)
            if chunk is None:
                issues.append(EvidenceValidationEntry(
                    claim=item.claim,
                    chunk_id=source.chunk_id,
                    issue=f"Cannot validate excerpt: chunk '{source.chunk_id}' not found",
                ))
                continue

            if not self._excerpt_matches(source.source_excerpt, chunk.text):
                issues.append(EvidenceValidationEntry(
                    claim=item.claim,
                    chunk_id=source.chunk_id,
                    issue="Source excerpt is not supported by the referenced chunk content",
                ))

        return issues

    @staticmethod
    def _excerpt_matches(excerpt: str, chunk_text: str) -> bool:
        """Check if *excerpt* is supported by *chunk_text*.

        Allows whitespace and line-ending normalisation.  Does *not*
        require exact byte-level equality.
        """
        if not excerpt:
            return True
        return _normalize_text(excerpt) in _normalize_text(chunk_text)

    # ── Unsupported claims ───────────────────────────────────────────────

    def _find_unsupported_claims(
        self,
        result: DocumentUnderstandingResult,
    ) -> list[UnsupportedClaimEntry]:
        """Identify domain-model values that lack supporting evidence."""
        unsupported: list[UnsupportedClaimEntry] = []

        evidence_claims = {item.claim for item in result.evidence}

        sections = {
            "project_profile": result.project_profile,
            "business": result.business,
            "goals": result.goals,
            "features": result.features,
            "requirements": result.requirements,
            "technical": result.technical,
            "other_info": result.other_info,
        }

        list_fields: dict[str, list[str]] = {
            "goals": ["goals", "objectives", "success_criteria"],
            "features": ["features", "core_features", "optional_features", "user_flows"],
            "requirements": ["functional_requirements", "non_functional_requirements",
                            "business_requirements", "technical_requirements",
                            "security_requirements"],
            "technical": ["technology_stack", "programming_languages", "frameworks",
                         "databases", "system_components", "integrations"],
            "other_info": ["constraints", "assumptions", "limitations", "dependencies",
                          "planned_features", "future_work"],
            "business": ["target_users", "stakeholders"],
        }

        for section_name, section in sections.items():
            if section is None:
                continue
            for field_name in list_fields.get(section_name, []):
                values = getattr(section, field_name, []) or []
                for v in values:
                    if v and str(v).strip() and str(v) not in evidence_claims:
                        unsupported.append(UnsupportedClaimEntry(
                            claim=str(v),
                            reason=f"'{section_name}.{field_name}' entry has no "
                                   f"corresponding evidence item",
                        ))

        return unsupported

    # ── Missing information validation ───────────────────────────────────

    def _validate_missing_information(
        self,
        result: DocumentUnderstandingResult,
    ) -> list[MissingInfoInconsistency]:
        """Check missing-information entries against available evidence."""
        inconsistencies: list[MissingInfoInconsistency] = []

        for info in result.missing_information:
            field_parts = self._extract_keywords(info.field_path)
            desc_keywords = self._extract_keywords(info.description)
            all_keywords = list(set(field_parts + desc_keywords))

            if not all_keywords:
                continue

            for ev in result.evidence:
                if any(kw.lower() in ev.claim.lower() for kw in all_keywords):
                    inconsistencies.append(MissingInfoInconsistency(
                        field_path=info.field_path,
                        description=info.description,
                        reason=f"Evidence claim references keywords from this "
                               f"missing-information entry: '{ev.claim[:100]}'",
                    ))
                    break

                excerpt = ev.source.source_excerpt or ""
                if any(kw.lower() in excerpt.lower() for kw in all_keywords):
                    inconsistencies.append(MissingInfoInconsistency(
                        field_path=info.field_path,
                        description=info.description,
                        reason=f"Evidence excerpt contains keywords from this "
                               f"missing-information entry: '{excerpt[:100]}'",
                    ))
                    break

        return inconsistencies

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from *text*."""
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", text)
        stopwords = {
            "the", "and", "for", "was", "not", "but", "are", "has", "had",
            "its", "can", "all", "any", "been", "could", "were", "would",
            "should", "have", "from", "this", "that", "which", "with",
            "what", "when", "where", "how", "about", "into", "over",
            "such", "each", "than", "then", "these", "those", "their",
            "them", "they", "will", "been", "also", "some", "does",
        }
        return [w for w in words if w.lower() not in stopwords]

    # ── Conflict validation ──────────────────────────────────────────────

    def _validate_conflicts(
        self,
        conflicts: list[Conflict],
    ) -> list[ConflictValidationEntry]:
        """Validate that conflicts have properly grounded evidence."""
        invalid: list[ConflictValidationEntry] = []

        for conflict in conflicts:
            if len(conflict.conflicting_claims) < 2:
                invalid.append(ConflictValidationEntry(
                    field_path=conflict.field_path,
                    issue="Conflict must have at least two conflicting claims",
                ))
                continue

            for cs in conflict.conflicting_claims:
                source = cs.source
                if source.chunk_id:
                    chunk = self._store.get_by_chunk_id(source.chunk_id)
                    if chunk is None:
                        invalid.append(ConflictValidationEntry(
                            field_path=conflict.field_path,
                            issue=f"Conflict claim references nonexistent "
                                  f"chunk '{source.chunk_id}'",
                        ))
                        break
                elif source.document_ref is not None:
                    doc_key = _ref_key(source.document_ref)
                    if doc_key is not None and self._context.document_ref_keys:
                        if doc_key not in self._context.document_ref_keys:
                            invalid.append(ConflictValidationEntry(
                                field_path=conflict.field_path,
                                issue=f"Conflict document_ref '{doc_key}' "
                                      f"not found in current request context",
                            ))
                            break
                else:
                    invalid.append(ConflictValidationEntry(
                        field_path=conflict.field_path,
                        issue="Conflict claim has no chunk_id or document_ref",
                    ))
                    break

        return invalid

    # ── README exclusion ─────────────────────────────────────────────────

    def _check_readme_exclusion(
        self,
        result: DocumentUnderstandingResult,
    ) -> list[str]:
        """Check that *result* does not contain README content."""
        warnings: list[str] = []

        for item in result.evidence:
            if _contains_readme_content(item.claim):
                warnings.append(
                    f"Evidence claim contains README-like content: "
                    f"'{item.claim[:80]}'"
                )
            if item.source.source_excerpt and _contains_readme_content(
                item.source.source_excerpt
            ):
                warnings.append(
                    f"Evidence excerpt contains README-like content: "
                    f"'{item.source.source_excerpt[:80]}'"
                )

        for conflict in result.conflicts:
            for cs in conflict.conflicting_claims:
                if _contains_readme_content(cs.claim):
                    warnings.append(
                        f"Conflict claim contains README-like content: "
                        f"'{cs.claim[:80]}'"
                    )

        domain_text_fields = [
            ("project_profile.title", result.project_profile.title if result.project_profile else None),
            ("project_profile.short_description", result.project_profile.short_description if result.project_profile else None),
            ("project_profile.detailed_description", result.project_profile.detailed_description if result.project_profile else None),
            ("business.problem_statement", result.business.problem_statement if result.business else None),
            ("business.business_context", result.business.business_context if result.business else None),
            ("business.value_proposition", result.business.value_proposition if result.business else None),
            ("technical.architecture", result.technical.architecture if result.technical else None),
            ("technical.authentication", result.technical.authentication if result.technical else None),
            ("technical.authorization", result.technical.authorization if result.technical else None),
            ("technical.deployment", result.technical.deployment if result.technical else None),
            ("technical.infrastructure", result.technical.infrastructure if result.technical else None),
            ("other_info.project_status", result.other_info.project_status if result.other_info else None),
        ]

        for name, value in domain_text_fields:
            if value and _contains_readme_content(value):
                warnings.append(f"'{name}' contains README-like content")

        list_sections = [
            ("goals.goals", result.goals.goals if result.goals else []),
            ("goals.objectives", result.goals.objectives if result.goals else []),
            ("features.features", result.features.features if result.features else []),
            ("features.core_features", result.features.core_features if result.features else []),
            ("requirements.functional_requirements", result.requirements.functional_requirements if result.requirements else []),
        ]

        for name, values in list_sections:
            for v in values:
                if _contains_readme_content(v):
                    warnings.append(f"'{name}' entry contains README-like content: '{v[:80]}'")

        return warnings

    # ── Aggregate status ────────────────────────────────────────────────

    def _set_aggregate_status(
        self,
        output: ValidationResult,
        result: DocumentUnderstandingResult,
    ) -> None:
        """Set the aggregate validation status fields."""
        has_invalid_evidence = bool(output.invalid_evidence)
        has_unsupported = bool(output.unsupported_claims)
        has_invalid_conflicts = bool(output.invalid_conflicts)
        has_errors = bool(output.validation_errors)

        if has_invalid_evidence:
            total_evidence = len(result.evidence)
            invalid_count = len(output.invalid_evidence)
            if invalid_count >= total_evidence > 0:
                output.evidence_validation_status = "invalid"
            else:
                output.evidence_validation_status = "partial"
        elif has_unsupported:
            output.evidence_validation_status = "partial"

        if has_errors:
            output.is_valid = False
        elif self._strict:
            output.is_valid = not (
                has_invalid_evidence or has_unsupported or has_invalid_conflicts
            )
        else:
            output.is_valid = True
