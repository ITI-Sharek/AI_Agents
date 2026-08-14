from __future__ import annotations

from sharek_agents.agents.document_understanding.agent import (
    AgentConfig,
    ReActAgent,
)
from sharek_agents.agents.document_understanding.chunker import (
    ChunkingConfig,
    DocumentChunk,
    DocumentChunker,
)
from sharek_agents.agents.document_understanding.cloudinary_client import (
    CloudinaryClientImpl,
    CloudinaryConfig,
    CloudinaryError,
)
from sharek_agents.agents.document_understanding.embeddings import (
    EmbeddingError,
    EmbeddingService,
    create_embedding_service,
)
from sharek_agents.agents.document_understanding.memory_store import (
    VectorStore,
)
from sharek_agents.agents.document_understanding.parser import (
    ParsedDocument,
    ParsingError,
    parse_document,
)
from sharek_agents.agents.document_understanding.prompts import SYSTEM_PROMPT
from sharek_agents.agents.document_understanding.registry import ToolRegistry
from sharek_agents.agents.document_understanding.schemas import (
    CloudinaryResourceRef,
    DocumentUnderstandingInput,
    DocumentUnderstandingResult,
    ValidationStatus,
)
from sharek_agents.agents.document_understanding.tools import (
    DocumentUnderstandingContext,
    GetChunkByIdTool,
    GetDocumentSectionTool,
    InspectDocumentTool,
    SearchProjectDocumentsTool,
)
from sharek_agents.agents.document_understanding.validator import (
    DocumentUnderstandingValidator,
    ValidationContext,
)
from sharek_agents.common.logging import get_logger
from sharek_agents.config import settings


logger = get_logger(__name__)


_MAX_DOCUMENTS = 20


class DocumentUnderstandingServiceError(Exception):
    """Base error for document understanding pipeline failures."""


class CloudinaryServiceError(DocumentUnderstandingServiceError):
    """Cloudinary document retrieval failed."""


class ParseServiceError(DocumentUnderstandingServiceError):
    """Document parsing failed."""


class EmbeddingServiceError(DocumentUnderstandingServiceError):
    """Embedding generation failed."""


class VectorStoreServiceError(DocumentUnderstandingServiceError):
    """Vector store operation failed."""


class AgentServiceError(DocumentUnderstandingServiceError):
    """LLM provider or agent execution error."""


class AgentServiceTimeout(DocumentUnderstandingServiceError):
    """Agent execution timed out or reached iteration limit."""


class ValidationServiceError(DocumentUnderstandingServiceError):
    """Output validation failed."""


def _ref_key(ref: CloudinaryResourceRef | None) -> str | None:
    if ref is None:
        return None
    return ref.public_id or ref.url or None


def _deduplicate_documents(
    documents: list[CloudinaryResourceRef],
) -> list[CloudinaryResourceRef]:
    seen: set[str] = set()
    result: list[CloudinaryResourceRef] = []
    for doc in documents:
        key = _ref_key(doc)
        if key is None:
            result.append(doc)
            continue
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


async def run_analysis_pipeline(
    body: DocumentUnderstandingInput,
) -> DocumentUnderstandingResult:
    vector_store = VectorStore()
    embedding_service: EmbeddingService | None = None
    context: DocumentUnderstandingContext | None = None

    try:
        documents = _deduplicate_documents(body.documents)

        if not documents:
            raise ValidationServiceError("At least one document reference is required")

        if len(documents) > _MAX_DOCUMENTS:
            raise ValidationServiceError(
                f"Maximum of {_MAX_DOCUMENTS} documents allowed, "
                f"got {len(documents)}"
            )

        logger.info(
            "Starting analysis for project=%s, documents=%d",
            body.project_id,
            len(documents),
        )

        cloudinary_config = CloudinaryConfig(
            cloud_name=settings.cloudinary_cloud_name,
            api_key=settings.cloudinary_api_key,
            api_secret=settings.cloudinary_api_secret,
            max_file_size_bytes=settings.cloudinary_max_file_size_bytes,
            default_resource_type=settings.cloudinary_default_resource_type,
            timeout_seconds=settings.doc_understanding_timeout_seconds,
        )
        cloudinary_client = CloudinaryClientImpl(config=cloudinary_config)

        retrieved_docs = []
        for doc_ref in documents:
            try:
                retrieved = await cloudinary_client.retrieve(doc_ref)
                retrieved_docs.append(retrieved)
                logger.debug(
                    "Retrieved document: %s (%d bytes)",
                    retrieved.filename,
                    retrieved.file_size,
                )
            except CloudinaryError as exc:
                raise CloudinaryServiceError(
                    f"Failed to retrieve document '{_ref_key(doc_ref)}': {exc}"
                ) from exc

        parsed_docs: list[ParsedDocument] = []
        for retrieved in retrieved_docs:
            try:
                parsed = await parse_document(
                    content=retrieved.content,
                    file_format=retrieved.file_format,
                    content_type=retrieved.content_type,
                    filename=retrieved.filename,
                    reference=retrieved.reference,
                )
                parsed_docs.append(parsed)
                logger.debug(
                    "Parsed document: %s (%d elements, %d chars)",
                    parsed.filename or "unknown",
                    len(parsed.elements),
                    len(parsed.text),
                )
            except ParsingError as exc:
                raise ParseServiceError(
                    f"Failed to parse document '{retrieved.filename}': {exc}"
                ) from exc

        chunk_config = ChunkingConfig(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            min_chunk_size=settings.chunk_min_size,
        )
        chunker = DocumentChunker(config=chunk_config)

        all_chunks: list[DocumentChunk] = []
        for parsed in parsed_docs:
            chunks = chunker.chunk_document(parsed)
            all_chunks.extend(chunks)
            logger.debug(
                "Chunked document '%s' into %d chunks",
                parsed.filename or "unknown",
                len(chunks),
            )

        if not all_chunks:
            raise ParseServiceError(
                "All documents are empty — no content to analyze"
            )

        embedding_service = create_embedding_service(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            timeout_seconds=settings.embedding_timeout_seconds,
        )

        try:
            texts = [chunk.text for chunk in all_chunks]
            embeddings = await embedding_service.embed_texts(texts)
            logger.debug(
                "Generated %d embeddings (dim=%d)",
                len(embeddings),
                embedding_service.dimensions,
            )
        except EmbeddingError as exc:
            raise EmbeddingServiceError(
                f"Failed to generate embeddings: {exc}"
            ) from exc

        try:
            vector_store.ingest(all_chunks, embeddings)
            logger.debug(
                "Indexed %d chunks in vector store",
                vector_store.size,
            )
        except Exception as exc:
            raise VectorStoreServiceError(
                f"Failed to index embeddings in vector store: {exc}"
            ) from exc

        context = DocumentUnderstandingContext(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        context.register_chunks(all_chunks)
        logger.debug(
            "Built analysis context with %d documents",
            context.document_count,
        )

        registry = ToolRegistry()
        search_tool = SearchProjectDocumentsTool(context)
        get_chunk_tool = GetChunkByIdTool(context)
        section_tool = GetDocumentSectionTool(context)
        inspect_tool = InspectDocumentTool(context)
        registry.register(search_tool)
        registry.register(get_chunk_tool)
        registry.register(section_tool)
        registry.register(inspect_tool)

        agent = ReActAgent()
        agent_config = AgentConfig(
            model=settings.doc_understanding_model,
            provider=settings.doc_understanding_provider,
            system_prompt=SYSTEM_PROMPT,
            tools=[search_tool, get_chunk_tool, section_tool, inspect_tool],
            max_iterations=10,
            project_id=body.project_id,
            timeout_seconds=settings.doc_understanding_timeout_seconds,
        )

        query = (
            "Analyze the provided project documentation and extract "
            "a comprehensive project profile."
        )
        document_text = (
            f"The following documents have been indexed and are available "
            f"for analysis. Project ID: {body.project_id}. "
            f"Number of documents: {len(documents)}. "
            f"Filenames: {', '.join(d.filename or 'unknown' for d in retrieved_docs)}. "
            f"Use the search tools to explore the content."
        )

        agent_result = await agent.run_with_metadata(query, document_text, agent_config)

        if agent_result.max_iterations_reached:
            raise AgentServiceTimeout("Agent reached maximum iteration limit")
        if not agent_result.completed_successfully:
            warnings = agent_result.result.validation_status.warnings or []
            joined = " ".join(warnings).lower()
            if "timed out" in joined:
                raise AgentServiceTimeout("Agent execution timed out")
            if "failed to parse" in joined or "provider error" in joined:
                raise AgentServiceError(
                    "LLM provider returned an invalid or unparseable response"
                )
            if "empty" in joined and not warnings:
                raise AgentServiceError(
                    "Agent returned an empty response"
                )

        result = agent_result.result
        doc_ref_keys: set[str] = set()
        for ref in documents:
            key = _ref_key(ref)
            if key:
                doc_ref_keys.add(key)

        validation_ctx = ValidationContext(
            vector_store=vector_store,
            document_ref_keys=doc_ref_keys,
        )
        validator = DocumentUnderstandingValidator(validation_ctx, strict=True)
        validation_result = validator.validate(result)

        if not validation_result.is_valid:
            errors = list(validation_result.validation_errors)
            if validation_result.invalid_evidence:
                for inv in validation_result.invalid_evidence:
                    errors.append(f"Invalid evidence: {inv.issue}")
            if validation_result.unsupported_claims:
                for uc in validation_result.unsupported_claims[:5]:
                    errors.append(f"Unsupported claim: {uc.claim[:100]}")
            raise ValidationServiceError(
                "; ".join(errors) if errors else "Output validation failed"
            )

        result.validation_status = ValidationStatus(
            is_valid=True,
            missing_required=[],
            warnings=validation_result.validation_warnings or [],
        )
        logger.info(
            "Analysis completed for project=%s: valid=%s, "
            "evidence=%d, missing=%d, conflicts=%d",
            body.project_id,
            validation_result.is_valid,
            len(result.evidence),
            len(result.missing_information),
            len(result.conflicts),
        )

        return result

    except (CloudinaryServiceError, ParseServiceError,
            EmbeddingServiceError, VectorStoreServiceError,
            AgentServiceError, AgentServiceTimeout,
            ValidationServiceError):
        raise

    except Exception as exc:
        logger.error(
            "Unexpected error in analysis pipeline for project=%s: %s",
            body.project_id,
            exc,
            exc_info=True,
        )
        raise DocumentUnderstandingServiceError(
            "An unexpected error occurred during document analysis"
        ) from exc

    finally:
        try:
            vector_store.clear()
            logger.debug("Vector store cleared")
        except Exception:
            pass
        if embedding_service is not None and hasattr(embedding_service, "close"):
            try:
                await embedding_service.close()
                logger.debug("Embedding service closed")
            except Exception:
                pass
