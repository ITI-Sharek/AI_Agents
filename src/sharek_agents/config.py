import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))


settings = Settings()
