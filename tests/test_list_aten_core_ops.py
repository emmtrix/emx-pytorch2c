import os
import subprocess
import sys
from pathlib import Path

REFERENCE_FILE = Path(__file__).resolve().parent / "list_aten_core_ops_ref.txt"


def _run_list_aten_core_ops() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "list_aten_core_ops.py"
    output = subprocess.check_output(
        [sys.executable, str(script_path)],
        text=True,
    )
    return output


def test_list_aten_core_ops_matches_reference():
    output = _run_list_aten_core_ops()

    if os.getenv("UPDATE_REFS"):
        REFERENCE_FILE.write_text(output, encoding="utf-8")

    expected = REFERENCE_FILE.read_text(encoding="utf-8")
    assert output == expected
