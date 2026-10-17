"""Typed application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Validated configuration for every external provider seam."""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    memmachine_base_url: AnyHttpUrl = AnyHttpUrl("https://api.memmachine.ai")
    memmachine_api_key: SecretStr | None = None
    memmachine_project_id: str = "recursive-agents-hackathon"
    memmachine_org_id: str | None = None

    crusoe_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://api.inference.crusoecloud.com/v1"
    )
    crusoe_api_key: SecretStr | None = None
    crusoe_student_model: str | None = None
    crusoe_reviewer_model: str | None = None
    crusoe_consolidator_model: str | None = None

    openshell_base_url: AnyHttpUrl | None = None
    openshell_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
