"""evals/run_suite.py reuses a cached config only for the same setup, and fills in missing row dumps."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(tmp_path, *extra):
    cmd = [sys.executable, str(ROOT / "evals/run_suite.py"), "--backend", "mock", "--model", "mock",
           "--configs", "boolq", "--out-dir", str(tmp_path / "out"), "--calibration-dir", str(tmp_path / "cal"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout


def test_suite_cache_respects_setup_and_row_dumps(tmp_path):
    first = _run(tmp_path, "--rows", "4")
    assert "cached" not in first
    assert "[boolq] cached\n" in _run(tmp_path, "--rows", "4")
    changed = _run(tmp_path, "--rows", "6")
    assert "setup changed" in changed and "[boolq] cached\n" not in changed
    result = json.loads((tmp_path / "out" / "mock.json").read_text())
    assert result["rows_per_split"] == 6 and result["configs"]["boolq"]["raw"]["count"] == 6
    dumped = _run(tmp_path, "--rows", "6", "--dump-rows")
    assert "[boolq] cached\n" not in dumped
    assert (tmp_path / "out" / "rows" / "mock" / "boolq.json").exists()
