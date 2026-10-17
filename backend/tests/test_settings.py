from decimal import Decimal
from pathlib import Path

from pydantic import SecretStr

from memory_distillation.main import (
    app,
    create_production_memory,
    create_production_sandbox,
)
from memory_distillation.operational_memory import MemMachineOperationalMemory
from memory_distillation.sandbox import OpenShellSandbox
from memory_distillation.settings import Settings, get_settings


def test_settings_load_every_supported_environment_variable(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "MEMMACHINE_BASE_URL=https://memory.example.test",
                "MEMMACHINE_API_KEY=memory-secret",
                "MEMMACHINE_PROJECT_ID=project-1",
                "CRUSOE_BASE_URL=https://inference.example.test/v1",
                "CRUSOE_API_KEY=crusoe-secret",
                "CRUSOE_STUDENT_MODEL=student-v1",
                "CRUSOE_REVIEWER_MODEL=reviewer-v1",
                "CRUSOE_CONSOLIDATOR_MODEL=consolidator-v1",
                "CRUSOE_HSTAR_MODEL=hstar-v1",
                "OPENSHELL_SANDBOX_NAME=memory-distill",
                "OPENSHELL_GATEWAY=nemoclaw-test",
            )
        )
        + "\n"
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert str(settings.memmachine_base_url) == "https://memory.example.test/"
    assert settings.memmachine_api_key == SecretStr("memory-secret")
    assert settings.memmachine_project_id == "project-1"
    assert str(settings.crusoe_base_url) == "https://inference.example.test/v1"
    assert settings.crusoe_api_key == SecretStr("crusoe-secret")
    assert settings.crusoe_student_model == "student-v1"
    assert settings.crusoe_reviewer_model == "reviewer-v1"
    assert settings.crusoe_consolidator_model == "consolidator-v1"
    assert settings.crusoe_hstar_model == "hstar-v1"
    assert settings.crusoe_max_cost_usd == Decimal("0.05")
    assert settings.crusoe_max_total_tokens == 500_000
    assert settings.crusoe_timeout_seconds == 60
    assert settings.crusoe_max_retries == 1
    assert settings.openshell_sandbox_name == "memory-distill"
    assert settings.openshell_gateway == "nemoclaw-test"
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
                "CRUSOE_HSTAR_MODEL=",
                "OPENSHELL_SANDBOX_NAME=",
                "OPENSHELL_GATEWAY=",
            )
        )
        + "\n"
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert str(settings.crusoe_base_url) == "https://api.inference.crusoecloud.com/v1"
    assert settings.crusoe_api_key is None
    assert settings.crusoe_student_model is None
    assert settings.crusoe_hstar_model is None
    assert settings.openshell_sandbox_name is None
    assert settings.openshell_gateway == "nemoclaw"


def test_application_uses_the_cached_settings_instance() -> None:
    assert get_settings() is get_settings()
    assert app.state.settings is get_settings()


def test_production_composition_uses_configured_openshell_sandbox(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENSHELL_SANDBOX_NAME=memory-distill\nOPENSHELL_GATEWAY=nemoclaw\n")

    sandbox = create_production_sandbox(Settings(_env_file=env_file))  # type: ignore[call-arg]

    assert isinstance(sandbox, OpenShellSandbox)


def test_production_composition_uses_configured_managed_memory(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEMMACHINE_API_KEY=fixture-secret\nMEMMACHINE_PROJECT_ID=fixture-project\n"
    )
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    memory = create_production_memory(settings)

    assert isinstance(memory, MemMachineOperationalMemory)
    memory.close()
