import importlib.util
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_count_test_ops():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "count_test_ops.py"
    return _load_module("count_test_ops", module_path)


def _load_test_ops():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tests" / "test_cref_ops.py"
    return _load_module("test_cref_ops", module_path)


def test_count_test_ops_runs():
    module = _load_count_test_ops()
    test_ops = _load_test_ops()
    counts = module._count_executed_tests()

    expected_ops = {op.name for op in test_ops.OPS_UNDER_TEST}
    assert set(counts) == expected_ops

    total_matches = 0
    for data in counts.values():
        assert set(data) == {"matches", "invalid"}
        assert isinstance(data["matches"], int)
        assert isinstance(data["invalid"], int)
        assert data["matches"] >= 0
        assert data["invalid"] == 0
        total_matches += data["matches"]

    assert total_matches > 0
