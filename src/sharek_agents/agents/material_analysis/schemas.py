from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/markdown",
    "text/x-markdown",
    "text/plain",
}
ALLOWED_PROJECT_FIELDS = {
    "title",
    "description",
    "technologies",
    "category",
    "difficulty",
}
ALLOWED_CATEGORIES = {"web", "mobile", "ai_ml", "devops", "tools_utilities"}
ALLOWED_DIFFICULTIES = {"beginner", "intermediate", "advanced"}


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class MaterialVersionReference(ContractModel):
    material_id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)


class MaterialVersionInput(MaterialVersionReference):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=150)
    content_base64: str = Field(min_length=1, max_length=80_000_000)

    @field_validator("material_id", "filename", "mime_type", "content_base64")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("mime_type")
    @classmethod
    def mime_type_is_supported(cls, value: str) -> str:
        normalized = value.split(";", 1)[0].strip().lower()
        if normalized not in ALLOWED_MIME_TYPES:
            raise ValueError("Material MIME type is not supported for analysis")
        return normalized


class MaterialAnalysisInput(ContractModel):
    analysis_run_id: str = Field(min_length=1, max_length=100)
    analysis_set_id: str = Field(min_length=1, max_length=100)
    project_id: str = Field(min_length=1, max_length=100)
    purpose: Literal["PROJECT_MATERIAL_DRAFTING"]
    materials: list[MaterialVersionInput] = Field(min_length=1, max_length=20)
    max_extracted_characters: int = Field(default=250_000, ge=1, le=1_000_000)
    contract_version: Literal["material-draft-v1"]

    @model_validator(mode="after")
    def material_versions_are_unique(self) -> "MaterialAnalysisInput":
        keys = [(item.material_id, item.version) for item in self.materials]
        if len(keys) != len(set(keys)):
            raise ValueError("Material versions must be unique within an Analysis Set")
        return self


class ProjectDraftSuggestion(ContractModel):
    target_field: Literal[
        "title", "description", "technologies", "category", "difficulty"
    ]
    value: str | list[str]
    rationale: str = Field(min_length=1, max_length=2_000)
    source_versions: list[MaterialVersionReference] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def value_matches_field(self) -> "ProjectDraftSuggestion":
        if self.target_field == "technologies":
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("technologies suggestions require a non-empty list")
            if any(not item.strip() or len(item) > 100 for item in self.value):
                raise ValueError("technology suggestions must be bounded non-blank strings")
        else:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("project text suggestions require a non-blank string")
            if self.target_field == "category" and self.value not in ALLOWED_CATEGORIES:
                raise ValueError("project category is not supported")
            if self.target_field == "difficulty" and self.value not in ALLOWED_DIFFICULTIES:
                raise ValueError("project difficulty is not supported")
        return self


class DraftRequirement(ContractModel):
    kind: Literal["required", "preferred"]
    text: str = Field(min_length=2, max_length=500)

    @field_validator("text")
    @classmethod
    def requirement_text_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("requirement text must not be blank")
        return value


class ContributionRequestDraftSuggestion(ContractModel):
    title: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=10, max_length=5_000)
    requirements: list[DraftRequirement] = Field(min_length=1, max_length=20)
    technology_tags: list[str] = Field(default_factory=list, max_length=20)
    difficulty: Literal["beginner", "intermediate", "advanced"] | None = None
    rationale: str = Field(min_length=1, max_length=2_000)
    source_versions: list[MaterialVersionReference] = Field(min_length=1, max_length=20)

    @field_validator("title", "description", "rationale")
    @classmethod
    def required_text_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("technology_tags")
    @classmethod
    def technology_tags_are_bounded(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > 50 for item in cleaned):
            raise ValueError("technology tags must be bounded non-blank strings")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("technology tags must be unique")
        return cleaned

    @model_validator(mode="after")
    def requires_required_requirement(self) -> "ContributionRequestDraftSuggestion":
        if not any(item.kind == "required" for item in self.requirements):
            raise ValueError("at least one required requirement is needed")
        return self


class MaterialDraftProviderOutput(ContractModel):
    project_suggestions: list[ProjectDraftSuggestion] = Field(default_factory=list, max_length=5)
    contribution_request_suggestions: list[ContributionRequestDraftSuggestion] = Field(
        default_factory=list, max_length=5
    )


class MaterialAnalysisMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=150)
    prompt_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int = Field(ge=0)
    document_count: int = Field(ge=1, le=20)
    extracted_characters: int = Field(ge=1, le=1_000_000)


class MaterialAnalysisChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=200)
    material_id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=10_000)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)
    embedding: list[float] = Field(min_length=1, max_length=4096)


class MaterialAnalysisResult(ContractModel):
    status: Literal["COMPLETED"]
    project_suggestions: list[ProjectDraftSuggestion] = Field(default_factory=list)
    contribution_request_suggestions: list[ContributionRequestDraftSuggestion] = Field(
        default_factory=list
    )
    metadata: MaterialAnalysisMetadata
    chunks: list[MaterialAnalysisChunk] = Field(default_factory=list, max_length=2000)
