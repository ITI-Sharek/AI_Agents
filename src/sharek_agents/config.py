import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _env(primary: str, legacy: str | None = None, default: str = "") -> str:
    """Read a canonical setting with an optional legacy fallback."""
    value = os.environ.get(primary)
    if value is None and legacy is not None:
        value = os.environ.get(legacy)
    return (value if value is not None else default).strip()


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
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: _env(
            "GROQ_MODEL", "LLM_MODEL", "openai/gpt-oss-120b"
        )
    )
    alibaba_api_key: str = field(
        default_factory=lambda: os.environ.get("ALIBABA_API_KEY", "")
    )
    alibaba_base_url: str = field(
        default_factory=lambda: os.environ.get("ALIBABA_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    )
    alibaba_model: str = field(
        default_factory=lambda: os.environ.get(
            "ALIBABA_MODEL", "qwen2.5-coder-32b-instruct"
        )
    )
    alibaba_enable_thinking: bool = field(
        default_factory=lambda: os.environ.get(
            "ALIBABA_ENABLE_THINKING", "false"
        ).lower()
        == "true"
    )
    default_model: str = field(
        default_factory=lambda: _env("DEFAULT_MODEL", "LLM_MODEL", "qwen2.5-coder-32b-instruct")
    )
    ai_provider: str = field(
        default_factory=lambda: _env(
            "AI_PROVIDER", "LLM_PROVIDER", "alibaba"
        ).lower()
    )
    ai_service_auth_token: str = field(
        default_factory=lambda: os.environ.get("AI_SERVICE_AUTH_TOKEN", "")
    )
    ai_skill_profile_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("AI_SKILL_PROFILE_TIMEOUT_SECONDS", "60")
        )
    )
    ai_skill_gap_guidance_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("AI_SKILL_GAP_GUIDANCE_TIMEOUT_SECONDS", "75")
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

    # Temporary compatibility path for the NestJS material-analysis contract.
    # It accepts selected material bytes from the backend and must remain
    # explicitly enabled in development only.
    material_analysis_dev_mode: bool = field(
        default_factory=lambda: os.environ.get(
            "MATERIAL_ANALYSIS_DEV_MODE", "false"
        ).lower()
        == "true"
    )
    material_analysis_dev_max_file_size_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("MATERIAL_ANALYSIS_DEV_MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
        )
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
    doc_understanding_output_language: str = field(
        default_factory=lambda: os.environ.get(
            "DOC_UNDERSTANDING_OUTPUT_LANGUAGE", "ar"
        )
    )

    # Skill Profiling Agent LLM (dedicated TokenRouter configuration).
    # Independent from the shared common LLM: the Skill Profiling Agent
    # resolves its provider, base URL, model, and API key only from these
    # settings (see ``skill_profiling_agent/llm.py``).
    skill_profiling_llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "SKILL_PROFILING_LLM_MODEL", "moonshotai/kimi-k3-free"
        )
    )
    skill_profiling_llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "SKILL_PROFILING_LLM_BASE_URL", "https://api.tokenrouter.com/v1"
        )
    )
    skill_profiling_llm_api_key: str = field(
        default_factory=lambda: os.environ.get("SKILL_PROFILING_LLM_API_KEY", "")
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

    # Gap Guidance Agent - dedicated LLM configuration. The Gap Guidance
    # Agent resolves its provider, model, API key, and base URL only from
    # these settings (see ``gap_guidance/llm.py``). When they are unset,
    # the agent falls back to the shared OpenRouter configuration
    # (``common.llm.get_llm``), preserving the previous behavior.
    gap_guidance_llm_provider: str = field(
        default_factory=lambda: os.environ.get("GAP_GUIDANCE_LLM_PROVIDER", "")
    )
    gap_guidance_llm_model: str = field(
        default_factory=lambda: os.environ.get("GAP_GUIDANCE_LLM_MODEL", "")
    )
    gap_guidance_llm_api_key: str = field(
        default_factory=lambda: os.environ.get("GAP_GUIDANCE_LLM_API_KEY", "")
    )
    gap_guidance_llm_base_url: str = field(
        default_factory=lambda: os.environ.get("GAP_GUIDANCE_LLM_BASE_URL", "")
    )

    # Roadmap RAG - dedicated embedding configuration. Roadmap RAG resolves
    # its embedding provider, model, API key, and base URL only from these
    # settings (see ``roadmap_rag/embeddings.py``). When they are unset, it
    # falls back to the existing Semantic Matching embedding configuration
    # (``semantic_matching/llm.py``), preserving the previous behavior.
    roadmap_rag_embedding_provider: str = field(
        default_factory=lambda: os.environ.get("ROADMAP_RAG_EMBEDDING_PROVIDER", "")
    )
    roadmap_rag_embedding_model: str = field(
        default_factory=lambda: os.environ.get("ROADMAP_RAG_EMBEDDING_MODEL", "")
    )
    roadmap_rag_embedding_api_key: str = field(
        default_factory=lambda: os.environ.get("ROADMAP_RAG_EMBEDDING_API_KEY", "")
    )
    roadmap_rag_embedding_base_url: str = field(
        default_factory=lambda: os.environ.get("ROADMAP_RAG_EMBEDDING_BASE_URL", "")
    )

    # Advisory Fit - dedicated LLM configuration. The Advisory Fit agent
    # resolves its provider, model, API key, and base URL only from these
    # settings (see ``advisory_fit/llm.py``). When they are unset, it falls
    # back to the shared OpenRouter configuration (``common.llm.get_llm``),
    # preserving the previous behavior.
    advisory_fit_llm_provider: str = field(
        default_factory=lambda: os.environ.get("ADVISORY_FIT_LLM_PROVIDER", "")
    )
    advisory_fit_llm_model: str = field(
        default_factory=lambda: os.environ.get("ADVISORY_FIT_LLM_MODEL", "")
    )
    advisory_fit_llm_api_key: str = field(
        default_factory=lambda: os.environ.get("ADVISORY_FIT_LLM_API_KEY", "")
    )
    advisory_fit_llm_base_url: str = field(
        default_factory=lambda: os.environ.get("ADVISORY_FIT_LLM_BASE_URL", "")
    )

    # Semantic Matching - independent matching index (PostgreSQL + pgvector)
    semantic_matching_database_url: str = field(
        default_factory=lambda: os.environ.get("SEMANTIC_MATCHING_DATABASE_URL", "")
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

    @property
    def active_chat_model(self) -> str:
        """Return the configured model for the active chat provider."""
        if self.ai_provider == "alibaba":
            return self.alibaba_model
        if self.ai_provider == "openrouter":
            return self.openrouter_model
        if self.ai_provider == "groq":
            return self.groq_model
        return self.default_model


settings = Settings()
