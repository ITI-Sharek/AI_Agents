from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from sharek_agents.agents.document_understanding.parser import (
    DocumentElement,
    ParsedDocument,
)
from sharek_agents.agents.document_understanding.schemas import CloudinaryResourceRef


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Unique identifier for this chunk within the request scope")
    text: str = Field(description="Chunk text content")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata (page number, offset, etc.)")

    document_reference: CloudinaryResourceRef | None = Field(
        default=None, description="Original Cloudinary resource reference"
    )
    filename: str | None = Field(default=None, description="Source filename")
    file_format: str | None = Field(default=None, description="File format extension")
    page_number: int | None = Field(default=None, ge=1, description="Page number within source document")
    section: str | None = Field(default=None, description="Section or heading context as a path")
    character_start: int | None = Field(default=None, ge=0, description="Character offset in source document text")
    character_end: int | None = Field(default=None, ge=0, description="Character offset end (exclusive) in source document text")


class Chunker(Protocol):
    """Interface for splitting document text into overlapping chunks.

    Chunk size and overlap are configurable per implementation.
    """

    def chunk(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[DocumentChunk]:
        """Split text into a list of overlapping chunks.

        Args:
            text: The full document text.
            chunk_size: Target characters per chunk.
            overlap: Character overlap between consecutive chunks.

        Returns:
            An ordered list of DocumentChunk instances.
        """
        ...


@dataclass
class ChunkingConfig:
    """Configuration for the structure-aware document chunker.

    Parameters are intentionally mutable so callers can override
    individual values without rebuilding the config object.
    """
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 100


_SENTENCE_PATTERN = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")


def _stable_doc_id(doc: ParsedDocument) -> str:
    """Derive a deterministic, short document identifier."""
    if doc.filename:
        source = doc.filename
    elif doc.reference and doc.reference.public_id:
        source = doc.reference.public_id
    elif doc.reference and doc.reference.url:
        source = doc.reference.url
    else:
        source = hashlib.md5(doc.text.encode()).hexdigest()
    raw = hashlib.md5(source.encode()).hexdigest()[:12]
    return f"doc_{raw}"


def _assign_heading_paths(elements: list[DocumentElement]) -> list[list[str]]:
    """Return a heading-path list parallel to *elements*.

    Each entry is the list of heading texts from the document root
    to the innermost section enclosing that element.
    """
    stack: list[tuple[int, str]] = []
    paths: list[list[str]] = []

    for elem in elements:
        if elem.element_type == "heading" and elem.level is not None:
            while stack and stack[-1][0] >= elem.level:
                stack.pop()
            stack.append((elem.level, elem.text))
        paths.append([h[1] for h in stack])

    return paths


def _current_section(heading_path: list[str]) -> str | None:
    """Collapse a heading path into a single section string."""
    if not heading_path:
        return None
    return " / ".join(heading_path)


def _find_section_boundary(elements: list[DocumentElement], start: int) -> bool:
    """Return True if *start* is at a level-1 heading."""
    if start >= len(elements):
        return False
    e = elements[start]
    return e.element_type == "heading" and e.level == 1


def _find_heading(elements: list[DocumentElement], start: int) -> bool:
    """Return True if *start* is at any heading."""
    if start >= len(elements):
        return False
    return elements[start].element_type == "heading"


def _iter_char_offsets(
    elements: list[DocumentElement],
    full_text: str,
) -> list[tuple[int, int]]:
    """Compute (char_start, char_end) for each element in *full_text*."""
    offsets: list[tuple[int, int]] = []
    search_pos = 0
    for elem in elements:
        t = elem.text
        if not t:
            offsets.append((search_pos, search_pos))
            continue
        pos = full_text.find(t, search_pos)
        if pos >= 0:
            start = pos
            end = pos + len(t)
            search_pos = end
        else:
            start = search_pos
            end = search_pos + len(t)
            search_pos = end
        offsets.append((start, end))
    return offsets


def _split_on_sentences(text: str) -> list[str]:
    """Split *text* into sentences, preserving punctuation."""
    parts = _SENTENCE_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split_text(text: str, max_size: int) -> list[str]:
    """Split *text* at word boundaries when *max_size* is exceeded."""
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_size:
            parts.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_size)
        if split_at <= max_size // 2:
            split_at = remaining.find(" ", max_size)
            if split_at < 0:
                parts.append(remaining)
                break
        parts.append(remaining[:split_at])
        remaining = remaining[split_at:].strip()
    return parts


def _primary_page(elements: list[DocumentElement]) -> int | None:
    """Return the most common page number among *elements*."""
    pages = [e.page_number for e in elements if e.page_number is not None]
    if not pages:
        return None
    return max(set(pages), key=pages.count)


def _element_text_len(elements: list[DocumentElement]) -> int:
    return sum(len(e.text) for e in elements)


_PLACEHOLDER_ORDER = 0


class DocumentChunker:
    """Structure-aware document chunker that splits ``ParsedDocument``
    instances into retrieval-ready chunks while preserving source
    traceability.

    The chunker respects document structure boundaries in this order:
    sections (h1), headings (h2+), paragraphs, sentences.

    It is fully deterministic: the same ``ParsedDocument`` and
    ``ChunkingConfig`` always produce identical chunk outputs and IDs.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    # ── Public API ─────────────────────────────────────────────────────

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Convert a ``ParsedDocument`` into a list of ``DocumentChunk``s.

        Args:
            doc: A parsed document with structured elements.

        Returns:
            An ordered list of chunks preserving source order.
        """
        if not doc.elements or not doc.text.strip():
            return []

        heading_paths = _assign_heading_paths(doc.elements)
        char_offsets = _iter_char_offsets(doc.elements, doc.text)
        doc_id = _stable_doc_id(doc)

        units = self._build_units(doc.elements, heading_paths, char_offsets)
        chunks = self._assemble_chunks(units, doc, doc_id)
        chunks = self._apply_overlap(chunks)

        return [c for c in chunks if c.text.strip()]

    # ── Unit building ─────────────────────────────────────────────────

    def _build_units(
        self,
        elements: list[DocumentElement],
        heading_paths: list[list[str]],
        char_offsets: list[tuple[int, int]],
    ) -> list[dict]:
        """Convert elements into internal unit dicts, splitting oversized
        paragraphs at sentence boundaries when necessary."""
        units: list[dict] = []

        for i, elem in enumerate(elements):
            heading_path = heading_paths[i] if i < len(heading_paths) else []
            char_start, char_end = char_offsets[i] if i < len(char_offsets) else (0, 0)
            text = elem.text
            section = _current_section(heading_path)
            add_headings = elem.element_type == "heading"

            if add_headings:
                units.append(self._make_unit(elem, text, heading_path, char_start, char_end))
                continue

            split_needed = False
            if elem.element_type in ("paragraph", "list_item", "code_block", "table"):
                if len(text) > self.config.chunk_size:
                    split_needed = True

            if not split_needed:
                units.append(self._make_unit(elem, text, heading_path, char_start, char_end))
                continue

            sentences = _split_on_sentences(text)
            if len(sentences) > 1:
                per_sentence_offset = char_end - char_start
                if per_sentence_offset > 0:
                    ratio = per_sentence_offset / max(len(text), 1)
                else:
                    ratio = 0
                running = char_start
                for sent in sentences:
                    s_start = running
                    s_end = running + int(len(sent) * ratio) if ratio > 0 else running + len(sent)
                    sub = DocumentElement(
                        element_type=elem.element_type,
                        text=sent,
                        level=elem.level,
                        page_number=elem.page_number,
                        order=_PLACEHOLDER_ORDER,
                    )
                    units.append(self._make_unit(sub, sent, heading_path, s_start, s_end))
                    running = s_end
            else:
                parts = _hard_split_text(text, self.config.chunk_size)
                if not parts:
                    continue
                per_part = (char_end - char_start) / max(len(text), 1)
                running = char_start
                for part in parts:
                    p_start = running
                    p_end = running + int(len(part) * per_part)
                    sub = DocumentElement(
                        element_type=elem.element_type,
                        text=part,
                        level=elem.level,
                        page_number=elem.page_number,
                        order=_PLACEHOLDER_ORDER,
                    )
                    units.append(self._make_unit(sub, part, heading_path, p_start, p_end))
                    running = p_end

        return units

    def _make_unit(
        self,
        elem: DocumentElement,
        text: str,
        heading_path: list[str],
        char_start: int,
        char_end: int,
    ) -> dict:
        return {
            "element": elem,
            "text": text,
            "heading_path": heading_path,
            "section": _current_section(heading_path),
            "char_start": char_start,
            "char_end": char_end,
        }

    # ── Chunk assembly ────────────────────────────────────────────────

    def _assemble_chunks(
        self,
        units: list[dict],
        doc: ParsedDocument,
        doc_id: str,
    ) -> list[DocumentChunk]:
        """Greedily group units into chunks respecting structure boundaries."""
        chunks: list[DocumentChunk] = []
        acc: list[dict] = []
        acc_raw = 0

        def _joined_len(raw_sum: int, count: int) -> int:
            if count <= 0:
                return 0
            return raw_sum + (count - 1) * 2

        for unit in units:
            elem = unit["element"]
            text = unit["text"]
            is_heading = elem.element_type == "heading"
            heading_level = elem.level if is_heading else None
            is_section = is_heading and heading_level == 1
            current_joined = _joined_len(acc_raw, len(acc))
            would_join = _joined_len(acc_raw + len(text), len(acc) + 1)

            should_split = False

            if is_section and acc:
                should_split = True
            elif is_heading and heading_level is not None and heading_level > 1 and acc:
                if current_joined >= self.config.min_chunk_size:
                    should_split = True
                elif would_join > self.config.chunk_size:
                    should_split = True

            if should_split:
                chunks.append(self._finalize(acc, doc, doc_id, len(chunks)))
                acc = []
                acc_raw = 0
                current_joined = 0
                would_join = _joined_len(len(text), 1)

            if would_join > self.config.chunk_size and not is_heading and acc:
                chunks.append(self._finalize(acc, doc, doc_id, len(chunks)))
                acc = []
                acc_raw = 0

            acc.append(unit)
            acc_raw += len(text)

        if acc:
            chunks.append(self._finalize(acc, doc, doc_id, len(chunks)))

        return chunks

    def _finalize(
        self,
        units: list[dict],
        doc: ParsedDocument,
        doc_id: str,
        chunk_index: int,
    ) -> DocumentChunk:
        if not units:
            return DocumentChunk(chunk_id=f"{doc_id}_chunk_{chunk_index:04d}", text="")

        texts = [u["text"] for u in units]
        text = "\n\n".join(texts)

        elements = [u["element"] for u in units]
        page_number = _primary_page(elements)
        sections = [u.get("section") for u in units if u.get("section") is not None]
        section = sections[-1] if sections else None
        char_start = units[0]["char_start"]
        char_end = units[-1]["char_end"]

        chunk_id = f"{doc_id}_chunk_{chunk_index:04d}"

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            document_reference=doc.reference,
            filename=doc.filename,
            file_format=doc.file_format,
            page_number=page_number,
            section=section,
            character_start=char_start,
            character_end=char_end,
            metadata={
                "element_count": len(elements),
                "element_types": [e.element_type for e in elements],
                "element_orders": [e.order for e in elements],
            },
        )

    # ── Overlap ────────────────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Apply configurable overlap between consecutive chunks.

        Overlap is skipped when:
        - chunks belong to different sections (natural boundary)
        - the previous chunk is too small (< 2 * overlap)
        - overlap would exceed 50 % of the previous chunk
        """
        if self.config.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[DocumentChunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            prev_text = prev.text

            skip = (
                prev.section != curr.section
                or len(prev_text) < self.config.overlap * 2
                or (len(prev_text) > 0 and self.config.overlap / len(prev_text) > 0.5)
            )

            if skip:
                result.append(curr)
                continue

            overlap_text = prev_text[-self.config.overlap:]
            clean = _find_overlap_boundary(overlap_text)
            if clean >= 0:
                overlap_text = overlap_text[clean:]

            if overlap_text.strip():
                new_text = overlap_text + "\n\n" + curr.text
                new_char_start = max(0, (curr.character_start or 0) - len(overlap_text))
                result.append(
                    curr.model_copy(update={
                        "text": new_text,
                        "character_start": new_char_start,
                    })
                )
            else:
                result.append(curr)

        return result


def _find_overlap_boundary(text: str) -> int:
    """Return the index of the last sentence boundary in *text*, or -1."""
    for pattern in (". ", ".\n", "?\n", "!\n", ")\n", "\n\n"):
        idx = text.rfind(pattern)
        if idx >= 0:
            return idx + len(pattern)
    return -1
