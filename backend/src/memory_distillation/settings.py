"""Typed application configuration loaded from environment variables and `.env`."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr
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

    crusoe_base_url: AnyHttpUrl = AnyHttpUrl("https://api.inference.crusoecloud.com/v1")
    crusoe_api_key: SecretStr | None = None
    crusoe_student_model: str | None = None
    crusoe_reviewer_model: str | None = None
    crusoe_consolidator_model: str | None = None
    crusoe_max_cost_usd: Decimal = Field(default=Decimal("0.05"), gt=0)
    crusoe_max_total_tokens: int = Field(default=500_000, gt=0)
    crusoe_timeout_seconds: float = Field(default=60, gt=0)
    crusoe_max_retries: int = Field(default=1, ge=0, le=3)

    openshell_base_url: AnyHttpUrl | None = None
    openshell_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
