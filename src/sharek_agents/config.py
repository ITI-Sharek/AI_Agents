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


settings = Settings()
