from __future__ import annotations

from fastapi import HTTPException, status

from sharek_agents.agents.document_understanding.orchestrator import (
    AgentServiceError,
    AgentServiceTimeout,
    CloudinaryServiceError,
    DocumentUnderstandingServiceError,
    EmbeddingServiceError,
    ParseServiceError,
    ValidationServiceError,
    VectorStoreServiceError,
    run_analysis_pipeline,
)
from sharek_agents.agents.document_understanding.schemas import (
    DocumentUnderstandingInput,
    DocumentUnderstandingResult,
)
from sharek_agents.common.logging import get_logger


logger = get_logger(__name__)


async def analyze_document(
    body: DocumentUnderstandingInput,
) -> DocumentUnderstandingResult:
    try:
        return await run_analysis_pipeline(body)
    except CloudinaryServiceError as exc:
        logger.warning("Cloudinary retrieval error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document retrieval from Cloudinary failed",
        ) from exc
    except ParseServiceError as exc:
        logger.warning("Document parsing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document parsing failed: {exc}",
        ) from exc
    except EmbeddingServiceError as exc:
        logger.warning("Embedding generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding generation failed",
        ) from exc
    except VectorStoreServiceError as exc:
        logger.warning("Vector store error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document indexing failed",
        ) from exc
    except AgentServiceTimeout as exc:
        logger.warning("Agent timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Document analysis timed out",
        ) from exc
    except AgentServiceError as exc:
        logger.warning("Agent execution error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document analysis provider error",
        ) from exc
    except ValidationServiceError as exc:
        logger.warning("Output validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Output validation failed: {exc}",
        ) from exc
    except DocumentUnderstandingServiceError as exc:
        logger.error("Internal pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during document analysis",
        ) from exc
