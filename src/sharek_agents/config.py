import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class Settings:
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
        )
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get(
            "DEFAULT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
        )
    )
    ai_provider: str = field(
        default_factory=lambda: os.environ.get("AI_PROVIDER", "openrouter")
    )
    ai_service_auth_token: str = field(
        default_factory=lambda: os.environ.get("AI_SERVICE_AUTH_TOKEN", "")
    )
    ai_skill_profile_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("AI_SKILL_PROFILE_TIMEOUT_SECONDS", "60")
        )
    )
    ai_advisory_fit_max_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("AI_ADVISORY_FIT_MAX_RETRIES", "1")
        )
    )
    service_version: str = field(
        default_factory=lambda: os.environ.get("AI_SERVICE_VERSION", "0.1.0")
    )

    getaway_iti_key: str = field(
        default_factory=lambda: os.environ.get("GETAWAY_ITI_KEY", "")
    )

    getaway_base_url: str = field(
        default_factory=lambda: os.environ.get("GETAWAY_BASE_URL", "http://apiaccess.iti.net.eg")
    )

    getaway_model: str = field(
        default_factory=lambda: os.environ.get("GETAWAY_MODEL", "antropic.claude-sonnet-4.6")
    )
    

    # Analysis service (REST API) settings
    analysis_service_enabled: bool = field(
        default_factory=lambda: os.environ.get("ANALYSIS_SERVICE_ENABLED", "true").lower() == "true"
    )
    analysis_service_url: str = field(
        default_factory=lambda: os.environ.get("ANALYSIS_SERVICE_URL", "http://127.0.0.1:8000")
    )
    analysis_service_auth_token: str = field(
        default_factory=lambda: os.environ.get("ANALYSIS_SERVICE_AUTH_TOKEN", "")
    )
    analysis_service_timeout: int = field(
        default_factory=lambda: int(os.environ.get("ANALYSIS_SERVICE_TIMEOUT", "190"))
    )

    # Document Understanding Agent settings
    doc_understanding_model: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_MODEL", "openai/gpt-4o"
        )
    )
    doc_understanding_provider: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_PROVIDER", "openrouter"
        )
    )

    doc_understanding_llm_provider: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_LLM_PROVIDER", "moonshot"
        )
    )
    doc_understanding_llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_LLM_MODEL", "kimi-k3"
        )
    )
    doc_understanding_llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_LLM_BASE_URL", "https://api.moonshot.ai/v1"
        )
    )
    doc_understanding_llm_api_key: str = field(
        default_factory=lambda: os.environ.get("DOC_UNDERSTANDING_LLM_API_KEY", "")
    )

    doc_understanding_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("DOC_UNDERSTANDING_TIMEOUT_SECONDS", "120")
        )
    )
    cloudinary_cloud_name: str = field(
        default_factory=lambda: os.environ.get("CLOUDINARY_CLOUD_NAME", "")
    )
    cloudinary_api_key: str = field(
        default_factory=lambda: os.environ.get("CLOUDINARY_API_KEY", "")
    )
    cloudinary_api_secret: str = field(
        default_factory=lambda: os.environ.get("CLOUDINARY_API_SECRET", "")
    )
    cloudinary_max_file_size_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("CLOUDINARY_MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024))
        )
    )
    cloudinary_default_resource_type: str = field(
        default_factory=lambda: os.environ.get(
            "CLOUDINARY_DEFAULT_RESOURCE_TYPE", "raw"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_EMBEDDING_MODEL",
            "nvidia/nemotron-3-embed-1b:free",
        )
    )
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_EMBEDDING_PROVIDER", "openrouter"
        )
    )
    embedding_dimensions: int | None = field(
        default_factory=lambda: (
            int(v)
            if (v := os.environ.get("DOC_UNDERSTANDING_EMBEDDING_DIMENSIONS")) is not None
            else None
        )
    )
    embedding_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("DOC_UNDERSTANDING_EMBEDDING_TIMEOUT", "30")
        )
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.environ.get("DOC_UNDERSTANDING_EMBEDDING_API_KEY", "")
    )
    doc_understanding_embedding_base_url: str = field(
        default_factory=lambda: os.environ.get("DOC_UNDERSTANDING_EMBEDDING_BASE_URL", "")
    )

    # Chunking
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "200"))
    )
    chunk_min_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_MIN_SIZE", "100"))
    )


settings = Settings()
