from decimal import Decimal
from pathlib import Path

from pydantic import SecretStr

from memory_distillation.main import app
from memory_distillation.settings import Settings, get_settings


def test_settings_load_every_supported_environment_variable(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "MEMMACHINE_BASE_URL=https://memory.example.test",
                "MEMMACHINE_API_KEY=memory-secret",
                "MEMMACHINE_PROJECT_ID=project-1",
                "MEMMACHINE_ORG_ID=org-1",
                "CRUSOE_BASE_URL=https://inference.example.test/v1",
                "CRUSOE_API_KEY=crusoe-secret",
                "CRUSOE_STUDENT_MODEL=student-v1",
                "CRUSOE_REVIEWER_MODEL=reviewer-v1",
                "CRUSOE_CONSOLIDATOR_MODEL=consolidator-v1",
                "CRUSOE_MAX_COST_USD=0.05",
                "CRUSOE_MAX_TOTAL_TOKENS=500000",
                "CRUSOE_TIMEOUT_SECONDS=60",
                "CRUSOE_MAX_RETRIES=1",
                "OPENSHELL_BASE_URL=https://sandbox.example.test",
                "OPENSHELL_API_KEY=sandbox-secret",
            )
        )
        + "\n"
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert str(settings.memmachine_base_url) == "https://memory.example.test/"
    assert settings.memmachine_api_key == SecretStr("memory-secret")
    assert settings.memmachine_project_id == "project-1"
    assert settings.memmachine_org_id == "org-1"
    assert str(settings.crusoe_base_url) == "https://inference.example.test/v1"
    assert settings.crusoe_api_key == SecretStr("crusoe-secret")
    assert settings.crusoe_student_model == "student-v1"
    assert settings.crusoe_reviewer_model == "reviewer-v1"
    assert settings.crusoe_consolidator_model == "consolidator-v1"
    assert settings.crusoe_max_cost_usd == Decimal("0.05")
    assert settings.crusoe_max_total_tokens == 500_000
    assert settings.crusoe_timeout_seconds == 60
    assert settings.crusoe_max_retries == 1
    assert str(settings.openshell_base_url) == "https://sandbox.example.test/"
    assert settings.openshell_api_key == SecretStr("sandbox-secret")
    assert "crusoe-secret" not in repr(settings)


def test_empty_optional_values_use_safe_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "MEMMACHINE_BASE_URL=https://api.memmachine.ai",
                "MEMMACHINE_PROJECT_ID=recursive-agents-hackathon",
                "CRUSOE_BASE_URL=",
                "CRUSOE_API_KEY=",
                "CRUSOE_STUDENT_MODEL=",
                "OPENSHELL_BASE_URL=",
                "OPENSHELL_API_KEY=",
            )
        )
        + "\n"
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert str(settings.crusoe_base_url) == "https://api.inference.crusoecloud.com/v1"
    assert settings.crusoe_api_key is None
    assert settings.crusoe_student_model is None
    assert settings.openshell_base_url is None
    assert settings.openshell_api_key is None


def test_application_uses_the_cached_settings_instance() -> None:
    assert get_settings() is get_settings()
    assert app.state.settings is get_settings()
