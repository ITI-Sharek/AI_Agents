from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Input models ──────────────────────────────────────────────────────────────


class CloudinaryResourceRef(BaseModel):
    """Reference to a Cloudinary resource.

    Does not assume the resource is always accessible via a public URL.
    At least one of ``public_id`` or ``url`` must be provided.
    """
    public_id: str | None = Field(default=None, description="Cloudinary public ID")
    resource_type: str | None = Field(default=None, description="Resource type: image, raw, video")
    delivery_type: str | None = Field(default=None, description="Delivery type: upload, private, authenticated, fetch, …")
    format: str | None = Field(default=None, description="File format extension, e.g. pdf, docx")
    url: str | None = Field(default=None, description="Public Cloudinary URL (if available)")
    mime_type: str | None = Field(default=None, description="MIME type hint, e.g. application/pdf")

    @field_validator("public_id")
    @classmethod
    def public_id_not_empty(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("public_id must not be empty")
        return value

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("url must not be empty")
        return value

    @field_validator("format")
    @classmethod
    def format_not_empty(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("format must not be empty")
        return value


class DocumentUnderstandingInput(BaseModel):
    """Request to analyse a set of Cloudinary documents and extract a Project Profile."""
    project_id: str = Field(min_length=1, description="Unique project identifier")
    documents: list[CloudinaryResourceRef] = Field(
        min_length=1,
        description="Cloudinary document references to analyse (at least one)",
    )

    @field_validator("documents")
    @classmethod
    def at_least_one_identifier(cls, value: list[CloudinaryResourceRef]) -> list[CloudinaryResourceRef]:
        for ref in value:
            if ref.public_id is None and ref.url is None:
                raise ValueError(
                    "Each document reference must supply at least one of public_id or url"
                )
        return value


# ── Output domain models ──────────────────────────────────────────────────────


class ProjectProfile(BaseModel):
    title: str | None = None
    short_description: str | None = None
    detailed_description: str | None = None


class Business(BaseModel):
    problem_statement: str | None = None
    business_context: str | None = None
    target_users: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    value_proposition: str | None = None


class Goals(BaseModel):
    goals: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class Features(BaseModel):
    features: list[str] = Field(default_factory=list)
    core_features: list[str] = Field(default_factory=list)
    optional_features: list[str] = Field(default_factory=list)
    user_flows: list[str] = Field(default_factory=list)


class Requirements(BaseModel):
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    business_requirements: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    security_requirements: list[str] = Field(default_factory=list)


class Technical(BaseModel):
    technology_stack: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    architecture: str | None = None
    system_components: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    authentication: str | None = None
    authorization: str | None = None
    deployment: str | None = None
    infrastructure: str | None = None


class OtherInfo(BaseModel):
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    project_status: str | None = None
    planned_features: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)


class EvidenceSourceRef(BaseModel):
    """Lightweight reference describing where a piece of evidence was found."""
    document_ref: CloudinaryResourceRef | None = None
    filename: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_id: str | None = None
    source_excerpt: str | None = None


class EvidenceItem(BaseModel):
    """A single claim extracted from a document together with its source trace."""
    claim: str = Field(min_length=1, description="The extracted claim or finding")
    source: EvidenceSourceRef = Field(
        default_factory=EvidenceSourceRef,
        description="Trace back to the originating document location",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How reliably the claim was stated in the source",
    )


class MissingInfo(BaseModel):
    """Information that was expected in the documents but could not be found."""
    field_path: str = Field(
        min_length=1,
        description="Dot-notation path to the missing section (e.g. 'technical.authentication')",
    )
    description: str = Field(min_length=1, description="What was expected but not found")
    searched_in: list[str] = Field(
        default_factory=list,
        description="Document filenames or identifiers that were searched",
    )


class ConflictSource(BaseModel):
    claim: str = Field(min_length=1, description="One side of the conflicting information")
    source: EvidenceSourceRef = Field(
        default_factory=EvidenceSourceRef,
        description="Where this side of the conflict was found",
    )


class Conflict(BaseModel):
    """Contradictory information found across documents or sections."""
    field_path: str = Field(
        min_length=1,
        description="Dot-notation path to the field with conflicting values",
    )
    conflicting_claims: list[ConflictSource] = Field(
        min_length=2,
        description="At least two conflicting claims with their sources",
    )
    description: str = Field(min_length=1, description="Human-readable explanation of the conflict")


class ValidationStatus(BaseModel):
    """Overall validation summary for the extracted profile."""
    is_valid: bool = Field(default=False, description="Whether the profile meets minimum completeness criteria")
    missing_required: list[str] = Field(
        default_factory=list,
        description="Required sections that are absent",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking concerns about the extracted data",
    )


class DocumentUnderstandingResult(BaseModel):
    """Complete result of a document understanding / Project Profile extraction."""
    project_id: str = Field(min_length=1)
    project_profile: ProjectProfile | None = None
    business: Business | None = None
    goals: Goals | None = None
    features: Features | None = None
    requirements: Requirements | None = None
    technical: Technical | None = None
    other_info: OtherInfo | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[MissingInfo] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    validation_status: ValidationStatus = Field(
        default_factory=lambda: ValidationStatus(is_valid=False),
    )


# ── Error model ───────────────────────────────────────────────────────────────


class AgentErrorInfo(BaseModel):
    code: Literal[
        "cloudinary_error",
        "parse_error",
        "provider_error",
        "timeout",
        "validation_error",
        "internal_error",
    ]
    message: str
