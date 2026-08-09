from __future__ import annotations

import asyncio
import base64
import binascii
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from pydantic import ValidationError

from sharek_agents.agents.document_understanding.parser import parse_document
from sharek_agents.agents.document_understanding.chunker import ChunkingConfig, DocumentChunker
from sharek_agents.agents.document_understanding.embeddings import EmbeddingService, create_embedding_service
from sharek_agents.agents.document_understanding.memory_store import VectorStore
from sharek_agents.agents.material_analysis.prompts import SYSTEM_PROMPT, render_prompt
from sharek_agents.agents.material_analysis.schemas import (
    MaterialAnalysisInput,
    MaterialAnalysisMetadata,
    MaterialAnalysisChunk,
    MaterialAnalysisResult,
    MaterialDraftProviderOutput,
)
from sharek_agents.common.llm import generate_structured, get_provider_metadata
from sharek_agents.config import settings


PROMPT_VERSION = "material-draft-v1"
SCHEMA_VERSION = "material-draft-v1"


class MaterialAnalysisInputError(Exception):
    """The backend sent content that cannot be safely analyzed."""


class MaterialAnalysisProviderError(Exception):
    """The configured provider failed or returned invalid draft output."""


class MaterialAnalysisProviderTimeout(MaterialAnalysisProviderError):
    """The provider exceeded the bounded analysis timeout."""


@dataclass(frozen=True)
class MaterialAnalysisProviderResponse:
    output: MaterialDraftProviderOutput
    provider: str
    model: str


Provider = Callable[
    [MaterialAnalysisInput, str],
    Awaitable[MaterialDraftProviderOutput | MaterialAnalysisProviderResponse],
]


@dataclass(frozen=True)
class ExtractedMaterial:
    material_id: str
    version: int
    filename: str
    text: str
    parsed: object


async def _invoke_provider(
    input_data: MaterialAnalysisInput,
    user_prompt: str,
) -> MaterialAnalysisProviderResponse:
    try:
        output = await asyncio.wait_for(
            generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=MaterialDraftProviderOutput,
            ),
            timeout=settings.material_analysis_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise MaterialAnalysisProviderTimeout(
            "Material analysis provider timed out"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise MaterialAnalysisProviderError(
            "Material analysis provider failed"
        ) from exc
    except Exception as exc:
        raise MaterialAnalysisProviderError(
            "Material analysis provider failed"
        ) from exc

    provider, model = get_provider_metadata()
    # The backend owns latency and the provider identity; the model output never
    # gets to claim a provider or a schema version of its own.
    return MaterialAnalysisProviderResponse(output=output, provider=provider, model=model)


def _decode_material(item) -> bytes:
    try:
        decoded = base64.b64decode(item.content_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise MaterialAnalysisInputError(
            "Material content is not valid base64"
        ) from exc
    if not decoded:
        raise MaterialAnalysisInputError("Material content is empty")
    return decoded


async def _extract_materials(input_data: MaterialAnalysisInput) -> tuple[list[ExtractedMaterial], int]:
    extracted: list[ExtractedMaterial] = []
    total_characters = 0
    for item in input_data.materials:
        content = _decode_material(item)
        try:
            parsed = await parse_document(
                content,
                content_type=item.mime_type,
                filename=item.filename,
            )
        except Exception as exc:
            raise MaterialAnalysisInputError(
                "A selected Material could not be parsed"
            ) from exc
        text = parsed.text.strip()
        if not text:
            raise MaterialAnalysisInputError("A selected Material contains no text")
        total_characters += len(text)
        if total_characters > input_data.max_extracted_characters:
            raise MaterialAnalysisInputError(
                "Selected Materials exceed the extracted text limit"
            )
        extracted.append(ExtractedMaterial(item.material_id, item.version, item.filename, text, parsed))
    return extracted, total_characters


async def _build_retrieval_context(
    materials: list[ExtractedMaterial],
) -> tuple[list[tuple[str, int, str, str]], list[MaterialAnalysisChunk]]:
    """Build an isolated chunk/vector index for this one analysis request."""
    chunker = DocumentChunker(ChunkingConfig())
    indexed = []
    for material in materials:
        for index, chunk in enumerate(chunker.chunk_document(material.parsed)):
            indexed.append(
                (
                    material,
                    chunk.model_copy(
                        update={
                            "chunk_id": f"{material.material_id}:{material.version}:{index}"
                        }
                    ),
                )
            )
    if not indexed:
        return [(m.material_id, m.version, m.filename, m.text) for m in materials], []

    embedding_service: EmbeddingService = create_embedding_service(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    texts = [chunk.text for _, chunk in indexed]
    embeddings = await embedding_service.embed_texts(texts)
    store = VectorStore()
    store.ingest([chunk for _, chunk in indexed], embeddings)
    query_embedding = await embedding_service.embed_query(
        "project facts, technologies, contribution opportunities, requirements"
    )
    matches = store.search(query_embedding, top_k=min(len(indexed), 12))
    by_id = {chunk.chunk_id: (material, chunk, embedding) for (material, chunk), embedding in zip(indexed, embeddings)}
    retrieved = []
    output_chunks = []
    for _, match in enumerate(matches):
        material, chunk, embedding = by_id[match.chunk_id]
        retrieved.append((material.material_id, material.version, material.filename, chunk.text))
        output_chunks.append(
            MaterialAnalysisChunk(
                chunk_id=chunk.chunk_id,
                material_id=material.material_id,
                version=material.version,
                text=chunk.text,
                character_start=chunk.character_start,
                character_end=chunk.character_end,
                embedding=embedding,
            )
        )
    return retrieved or [(m.material_id, m.version, m.filename, m.text) for m in materials], output_chunks


def _references_are_in_scope(
    output: MaterialDraftProviderOutput,
    input_data: MaterialAnalysisInput,
) -> None:
    allowed = {
        (item.material_id, item.version)
        for item in input_data.materials
    }
    suggestions = [
        *output.project_suggestions,
        *output.contribution_request_suggestions,
    ]
    for suggestion in suggestions:
        references = [
            (reference.material_id, reference.version)
            for reference in suggestion.source_versions
        ]
        if len(references) != len(set(references)):
            raise MaterialAnalysisProviderError(
                "Material analysis output contains duplicate provenance"
            )
        if any(reference not in allowed for reference in references):
            raise MaterialAnalysisProviderError(
                "Material analysis output cites a Material outside the Analysis Set"
            )


async def generate_material_analysis(
    input_data: MaterialAnalysisInput,
    *,
    provider: Provider | None = None,
) -> MaterialAnalysisResult:
    """Extract selected Materials and return one all-or-nothing draft result."""

    materials, extracted_characters = await _extract_materials(input_data)
    if provider is None:
        try:
            prompt_materials, chunks = await _build_retrieval_context(materials)
        except Exception as exc:
            raise MaterialAnalysisProviderError("Material retrieval preparation failed") from exc
    else:
        prompt_materials = [(m.material_id, m.version, m.filename, m.text) for m in materials]
        chunks = []
    user_prompt = render_prompt(prompt_materials, extracted_characters)
    started_at = time.perf_counter()
    try:
        raw_response = await (provider or _invoke_provider)(input_data, user_prompt)
    except (MaterialAnalysisProviderError, MaterialAnalysisInputError):
        raise
    except ValidationError as exc:
        raise MaterialAnalysisProviderError(
            "Material analysis provider returned invalid output"
        ) from exc
    except Exception as exc:
        raise MaterialAnalysisProviderError(
            "Material analysis provider failed"
        ) from exc

    try:
        if isinstance(raw_response, MaterialAnalysisProviderResponse):
            response = raw_response
        else:
            response = MaterialAnalysisProviderResponse(
                output=MaterialDraftProviderOutput.model_validate(raw_response),
                provider="deterministic-fake",
                model="deterministic-fake",
            )
        _references_are_in_scope(response.output, input_data)
    except ValidationError as exc:
        raise MaterialAnalysisProviderError(
            "Material analysis provider returned invalid output"
        ) from exc
    latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
    metadata = MaterialAnalysisMetadata(
        provider=response.provider,
        model=response.model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        service_version=settings.service_version,
        latency_ms=latency_ms,
        document_count=len(materials),
        extracted_characters=extracted_characters,
    )
    return MaterialAnalysisResult(
        status="COMPLETED",
        project_suggestions=response.output.project_suggestions,
        contribution_request_suggestions=response.output.contribution_request_suggestions,
        metadata=metadata,
        chunks=chunks,
    )
