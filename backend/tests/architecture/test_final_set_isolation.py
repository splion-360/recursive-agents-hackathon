"""Architecture guard for held-out ChartQA access."""

import ast
import importlib
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "memory_distillation"
ALLOWED_IMPORTER = SOURCE_ROOT / "evaluation" / "final.py"


def test_only_final_evaluation_composition_root_may_import_final_benchmark() -> None:
    """Production modules outside the final root cannot see the held-out loader."""

    offenders = _forbidden_final_imports(SOURCE_ROOT, allowed={ALLOWED_IMPORTER})

    assert offenders == []
    public_benchmark = importlib.import_module("memory_distillation.benchmark")
    assert not hasattr(public_benchmark, "FinalBenchmark")


def test_guard_detects_a_forbidden_import(tmp_path: Path) -> None:
    """Keep this architecture assertion from becoming a vacuous file scan."""

    source_root = tmp_path / "memory_distillation"
    source_root.mkdir()
    (source_root / "leak.py").write_text(
        "from memory_distillation.benchmark.final import FinalBenchmark\n",
        encoding="utf-8",
    )

    assert _forbidden_final_imports(source_root, allowed=set()) == [source_root / "leak.py"]


def _forbidden_final_imports(root: Path, *, allowed: set[Path]) -> list[Path]:
    offenders: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if path in allowed or (path.name == "final.py" and path.parent.name == "benchmark"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == (
                "memory_distillation.benchmark.final"
            ):
                offenders.append(path)
                break
            if isinstance(node, ast.Import) and any(
                alias.name == "memory_distillation.benchmark.final" for alias in node.names
            ):
                offenders.append(path)
                break
    return offenders
