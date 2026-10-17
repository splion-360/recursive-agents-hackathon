"""Typed application configuration loaded from environment variables and `.env`."""

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

    crusoe_base_url: AnyHttpUrl = AnyHttpUrl("https://api.inference.crusoecloud.com/v1")
    crusoe_api_key: SecretStr | None = None
    crusoe_student_model: str | None = None
    crusoe_reviewer_model: str | None = None
    crusoe_consolidator_model: str | None = None
    crusoe_hstar_model: str | None = None
    crusoe_max_total_tokens: int = Field(default=2_000_000, gt=0)
    crusoe_timeout_seconds: float = Field(default=60, gt=0)
    crusoe_max_retries: int = Field(default=1, ge=0, le=3)
    # Structured-output providers may spend substantial tokens satisfying a
    # schema even when the resulting action is short. Use each configured
    # boundary's supported ceiling so valid JSON is not truncated mid-object.
    crusoe_student_max_output_tokens: int = Field(default=4_096, ge=64, le=4_096)
    crusoe_reviewer_max_output_tokens: int = Field(default=4_096, ge=64, le=4_096)
    crusoe_hstar_max_output_tokens: int = Field(default=8_192, ge=128, le=8_192)

    openshell_sandbox_name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,18}$")
    openshell_gateway: str = Field(default="nemoclaw", min_length=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
