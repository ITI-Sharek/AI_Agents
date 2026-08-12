import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    ai_skill_gap_guidance_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("AI_SKILL_GAP_GUIDANCE_TIMEOUT_SECONDS", "60")
        )
    )
    ai_skill_gap_guidance_max_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("AI_SKILL_GAP_GUIDANCE_MAX_RETRIES", "1")
        )
    )
    ai_contributor_matching_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("AI_CONTRIBUTOR_MATCHING_TIMEOUT_SECONDS", "60")
        )
    )
    ai_contributor_matching_max_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("AI_CONTRIBUTOR_MATCHING_MAX_RETRIES", "1")
        )
    )
    service_version: str = field(
        default_factory=lambda: os.environ.get("AI_SERVICE_VERSION", "0.1.0")
    )


settings = Settings()
