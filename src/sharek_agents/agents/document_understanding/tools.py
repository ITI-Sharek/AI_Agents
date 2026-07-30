from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from sharek_agents.agents.document_understanding.chunker import DocumentChunk
from sharek_agents.agents.document_understanding.embeddings import (
    EmbeddingError,
    EmbeddingService,
)
from sharek_agents.agents.document_understanding.memory_store import (
    InMemoryStore,
    SearchError,
    SearchResult,
)
from sharek_agents.agents.document_understanding.schemas import CloudinaryResourceRef


ToolResultStatus = Literal["success", "validation_error", "execution_error", "not_found", "empty"]


class ToolInput(BaseModel):
    """Base class for typed tool input schemas.

    Tool implementations should subclass this to define their
    expected arguments as typed Pydantic fields.  The JSON Schema is
    auto-derived for LLM provider tool definitions.
    """


class ToolDefinition(BaseModel):
    """JSON Schema describing a callable tool for LLM native tool calling.

    This schema is serialised and passed to the LLM provider as part of
    the native function-calling API (e.g. OpenAI ``tools`` parameter).
    """
    name: str = Field(description="Unique tool name")
    description: str = Field(description="What the tool does")
    parameters: dict[str, Any] = Field(
        description="JSON Schema object describing valid arguments",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema describing the return value, when available",
    )


class NativeToolCall(BaseModel):
    """A structured tool call returned natively by the LLM.

    The provider parses the LLM response into these fields;
    no manual JSON or regex extraction is performed.
    """
    id: str = Field(description="Provider-assigned tool call identifier")
    name: str = Field(description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        description="Arguments parsed by the provider from the LLM response",
    )


class ToolResult(BaseModel):
    """Result of executing a single NativeToolCall.

    The ``status`` field distinguishes between execution outcomes
    so callers can handle failures, validation errors, and empty
    results without inspecting exception types.
    """
    tool_call_id: str = Field(
        default="",
        description="Matches NativeToolCall.id for correlation",
    )
    name: str = Field(default="", description="Tool name that was executed")
    status: ToolResultStatus = Field(
        default="success",
        description="Execution outcome",
    )
    output: str = Field(
        default="",
        description="String-serialised result returned to the LLM",
    )
    error_message: str = Field(
        default="",
        description="Safe, non-technical error description",
    )


class Tool(Protocol):
    """A callable tool with metadata for native LLM tool calling.

    Implementations must provide a ToolDefinition that the LLM
    provider can consume natively, and an ``execute`` method that
    receives the provider-parsed arguments directly.
    """

    @property
    def name(self) -> str:
        """Short unique identifier, must match ToolDefinition.name."""
        ...

    @property
    def definition(self) -> ToolDefinition:
        """Full JSON Schema definition for the LLM provider."""
        ...

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result.

        Args:
            **kwargs: Arguments already parsed by the provider
                      according to the ToolDefinition schema.

        Returns:
            A string representation of the result suitable for
            inclusion in the LLM conversation history.

        Raises:
            pydantic.ValidationError: If ``kwargs`` do not match
                the tool's input schema.  The caller (e.g. registry)
                is responsible for catching this and producing a
                structured ``ToolResult``.
            Exception: Any tool-specific exception.  The caller
                should catch these and produce safe error messages.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Document reference helpers
# ═══════════════════════════════════════════════════════════════════════════════

MAX_TOP_K = 50


def _doc_ref_key(
    document_reference: CloudinaryResourceRef | None,
    filename: str | None = None,
) -> str:
    """Derive a deterministic string key for a document reference.

    Uses the same approach as the chunker's ``_stable_doc_id`` so that
    the key is consistent across the request lifecycle.
    """
    if document_reference and document_reference.public_id:
        source = document_reference.public_id
    elif document_reference and document_reference.url:
        source = document_reference.url
    elif filename:
        source = filename
    else:
        return "doc_unknown"
    raw = hashlib.md5(source.encode()).hexdigest()[:12]
    return f"doc_{raw}"


def _filename_from_ref(ref: CloudinaryResourceRef | None) -> str | None:
    if ref is None:
        return None
    if ref.public_id:
        return ref.public_id.rsplit("/", 1)[-1]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Request-scoped context
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DocumentMetadata:
    """Safe metadata about a single indexed document.

    Returned by ``inspect_document``.  Contains structural information
    only — never the full document content.
    """
    doc_ref_key: str = ""
    filename: str | None = None
    file_format: str | None = None
    available_pages: list[int] = field(default_factory=list)
    available_sections: list[str] = field(default_factory=list)
    chunk_count: int = 0


class DocumentUnderstandingContext:
    """Request-scoped context providing tools with access to the
    embedding service, vector store, and document-level metadata.

    Each request creates its own context.  Contexts are never shared
    between requests.  All tools operate exclusively on the context
    they receive.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: InMemoryStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._documents: dict[str, DocumentMetadata] = {}
        self._chunks_by_doc: dict[str, list[DocumentChunk]] = {}
        self._doc_ref_map: dict[str, CloudinaryResourceRef | None] = {}

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def vector_store(self) -> InMemoryStore:
        return self._vector_store

    def register_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Register chunks and build document-level indexes.

        Called by the pipeline after chunks have been ingested into
        the vector store.  Maintains the document metadata and
        per-document chunk lists needed by ``inspect_document`` and
        ``get_document_section``.
        """
        doc_chunks: dict[str, list[DocumentChunk]] = {}
        for chunk in chunks:
            key = _doc_ref_key(chunk.document_reference, chunk.filename)
            if key not in doc_chunks:
                doc_chunks[key] = []
            doc_chunks[key].append(chunk)

        for ref_key, chunk_list in doc_chunks.items():
            first = chunk_list[0]
            pages = sorted(
                {c.page_number for c in chunk_list if c.page_number is not None}
            )
            seen: set[str] = set()
            ordered_sections: list[str] = []
            for c in chunk_list:
                s = c.section
                if s is not None and s not in seen:
                    seen.add(s)
                    ordered_sections.append(s)

            self._documents[ref_key] = DocumentMetadata(
                doc_ref_key=ref_key,
                filename=first.filename or _filename_from_ref(first.document_reference),
                file_format=first.file_format or "",
                available_pages=pages,
                available_sections=ordered_sections,
                chunk_count=len(chunk_list),
            )
            self._chunks_by_doc[ref_key] = chunk_list
            self._doc_ref_map[ref_key] = first.document_reference

    def get_document(self, doc_ref_key: str) -> DocumentMetadata | None:
        return self._documents.get(doc_ref_key)

    def get_document_chunks(self, doc_ref_key: str) -> list[DocumentChunk]:
        return self._chunks_by_doc.get(doc_ref_key, [])

    @property
    def document_count(self) -> int:
        return len(self._documents)


# ═══════════════════════════════════════════════════════════════════════════════
# SearchProjectDocumentsTool
# ═══════════════════════════════════════════════════════════════════════════════


class SearchProjectDocumentsInput(ToolInput):
    query: str = Field(description="Natural-language search query to find relevant document content")
    top_k: int = Field(default=5, description="Maximum number of results to return", ge=1, le=MAX_TOP_K)
    similarity_threshold: float = Field(
        default=0.0, description="Minimum cosine similarity threshold (0.0 to 1.0)", ge=0.0, le=1.0,
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()


class SearchProjectDocumentsTool:
    """Perform semantic search over the current request's indexed document chunks.

    Generates an embedding for the query, searches the request-scoped
    in-memory vector store, and returns the most relevant chunks with
    their source metadata.
    """

    def __init__(self, context: DocumentUnderstandingContext) -> None:
        self._context = context

    @property
    def name(self) -> str:
        return "search_project_documents"

    @property
    def description(self) -> str:
        return (
            "Search through the indexed project documents using semantic similarity. "
            "Returns relevant document chunks with their source metadata, "
            "similarity scores, and document references."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return SearchProjectDocumentsInput

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    async def execute(self, **kwargs: Any) -> str:
        validated = self.input_schema(**kwargs)
        return await self._run(validated)

    async def _run(self, args: SearchProjectDocumentsInput) -> str:
        query = args.query
        vector_store = self._context.vector_store
        embedding_service = self._context.embedding_service

        # Generate query embedding
        try:
            query_vector = await embedding_service.embed_query(query)
        except EmbeddingError as exc:
            return json.dumps({
                "status": "error",
                "error": f"Embedding generation failed: {exc}",
            })

        # Search the vector store
        try:
            results: list[SearchResult] = vector_store.search(
                query_embedding=query_vector,
                top_k=args.top_k,
                similarity_threshold=args.similarity_threshold,
            )
        except (SearchError, Exception) as exc:
            return json.dumps({
                "status": "error",
                "error": f"Search failed: {exc}",
            })

        if not results:
            return json.dumps({
                "status": "empty",
                "results": [],
                "message": "No matching document chunks found.",
            })

        output = []
        for sr in results:
            chunk = sr.chunk
            item = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.text,
                "similarity_score": round(sr.score, 4),
                "document_reference": _doc_ref_key(chunk.document_reference, chunk.filename),
                "filename": chunk.filename,
                "file_format": chunk.file_format,
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            output.append(item)

        return json.dumps({
            "status": "success",
            "results": output,
            "total_results": len(output),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# GetChunkByIdTool
# ═══════════════════════════════════════════════════════════════════════════════


class GetChunkByIdInput(ToolInput):
    chunk_id: str = Field(description="The unique identifier of the chunk to retrieve")

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be empty")
        return value.strip()


class GetChunkByIdTool:
    """Retrieve one exact chunk from the current request-scoped vector store by its ID.

    Does not perform semantic search.  Does not call embeddings or the LLM.
    """

    def __init__(self, context: DocumentUnderstandingContext) -> None:
        self._context = context

    @property
    def name(self) -> str:
        return "get_chunk_by_id"

    @property
    def description(self) -> str:
        return "Retrieve a single document chunk by its exact chunk ID."

    @property
    def input_schema(self) -> type[BaseModel]:
        return GetChunkByIdInput

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    async def execute(self, **kwargs: Any) -> str:
        validated = self.input_schema(**kwargs)
        return await self._run(validated)

    async def _run(self, args: GetChunkByIdInput) -> str:
        vector_store = self._context.vector_store
        chunk = vector_store.get_by_chunk_id(args.chunk_id)
        if chunk is None:
            return json.dumps({
                "status": "not_found",
                "error": f"Chunk with ID '{args.chunk_id}' not found.",
            })

        return json.dumps({
            "status": "success",
            "chunk": {
                "chunk_id": chunk.chunk_id,
                "content": chunk.text,
                "document_reference": _doc_ref_key(chunk.document_reference, chunk.filename),
                "filename": chunk.filename,
                "file_format": chunk.file_format,
                "page_number": chunk.page_number,
                "section": chunk.section,
            },
        })


# ═══════════════════════════════════════════════════════════════════════════════
# GetDocumentSectionTool
# ═══════════════════════════════════════════════════════════════════════════════


class GetDocumentSectionInput(ToolInput):
    document_reference: str = Field(description="Document reference string (returned by search or inspect)")
    section: str = Field(description="Section heading path to retrieve (e.g. 'Introduction' or 'Technical / Architecture')")
    page_number: int | None = Field(default=None, description="Optional page number filter", ge=1)

    @field_validator("document_reference")
    @classmethod
    def doc_ref_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_reference must not be empty")
        return value.strip()

    @field_validator("section")
    @classmethod
    def section_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section must not be empty")
        return value.strip()


class GetDocumentSectionTool:
    """Retrieve chunks belonging to a specific document section.

    Returns chunks in original document order.  Does not use semantic search.
    """

    def __init__(self, context: DocumentUnderstandingContext) -> None:
        self._context = context

    @property
    def name(self) -> str:
        return "get_document_section"

    @property
    def description(self) -> str:
        return (
            "Retrieve all document chunks that belong to a specific section "
            "within a document. Returns chunks in original document order."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return GetDocumentSectionInput

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    async def execute(self, **kwargs: Any) -> str:
        validated = self.input_schema(**kwargs)
        return await self._run(validated)

    async def _run(self, args: GetDocumentSectionInput) -> str:
        context = self._context
        doc_meta = context.get_document(args.document_reference)
        if doc_meta is None:
            return json.dumps({
                "status": "not_found",
                "error": f"Document '{args.document_reference}' not found.",
            })

        chunks = context.get_document_chunks(args.document_reference)
        matching = []
        for c in chunks:
            if c.section != args.section:
                continue
            if args.page_number is not None and c.page_number != args.page_number:
                continue
            matching.append(c)

        if not matching:
            return json.dumps({
                "status": "empty",
                "results": [],
                "message": f"No chunks found for section '{args.section}' in document '{args.document_reference}'.",
            })

        results = []
        for c in matching:
            results.append({
                "chunk_id": c.chunk_id,
                "content": c.text,
                "page_number": c.page_number,
                "section": c.section,
                "filename": c.filename,
                "document_reference": _doc_ref_key(c.document_reference, c.filename),
            })

        return json.dumps({
            "status": "success",
            "results": results,
            "total_results": len(results),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# InspectDocumentTool
# ═══════════════════════════════════════════════════════════════════════════════


class InspectDocumentInput(ToolInput):
    document_reference: str = Field(description="Document reference string (returned by search or inspect)")

    @field_validator("document_reference")
    @classmethod
    def doc_ref_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_reference must not be empty")
        return value.strip()


class InspectDocumentTool:
    """Provide a high-level structural overview of an indexed document.

    Returns safe metadata such as filename, file format, available pages,
    available sections, and chunk count.  Never returns full document content.
    """

    def __init__(self, context: DocumentUnderstandingContext) -> None:
        self._context = context

    @property
    def name(self) -> str:
        return "inspect_document"

    @property
    def description(self) -> str:
        return (
            "Get a high-level overview of an indexed document: filename, "
            "file format, available pages, sections, and chunk count."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return InspectDocumentInput

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    async def execute(self, **kwargs: Any) -> str:
        validated = self.input_schema(**kwargs)
        return await self._run(validated)

    async def _run(self, args: InspectDocumentInput) -> str:
        doc_meta = self._context.get_document(args.document_reference)
        if doc_meta is None:
            return json.dumps({
                "status": "not_found",
                "error": f"Document '{args.document_reference}' not found.",
            })

        return json.dumps({
            "status": "success",
            "document": {
                "document_reference": doc_meta.doc_ref_key,
                "filename": doc_meta.filename,
                "file_format": doc_meta.file_format,
                "available_pages": doc_meta.available_pages,
                "available_sections": doc_meta.available_sections,
                "chunk_count": doc_meta.chunk_count,
            },
        })
