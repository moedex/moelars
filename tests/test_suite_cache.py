"""evals/run_suite.py reuses a cached config only for the same setup, and fills in missing row dumps."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(tmp_path, *extra):
    cmd = [sys.executable, str(ROOT / "evals/run_suite.py"), "--backend", "mock", "--model", "mock",
           "--out-dir", str(tmp_path / "out"), "--calibration-dir", str(tmp_path / "cal"), *extra]
    if "--configs" not in extra:
        cmd += ["--configs", "boolq"]
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


def test_pooled_calibrator_is_fitted_once_and_used_for_every_config(tmp_path):
    cal = tmp_path / "pooled.json"
    out = _run(tmp_path, "--rows", "4", "--configs", "boolq,paws", "--calibration", str(cal))
    assert "pooled validation rows" in out and cal.exists()
    assert not list((tmp_path / "cal").glob("*.json"))  # no per-config calibrators
    again = _run(tmp_path, "--rows", "4", "--configs", "boolq,paws", "--calibration", str(cal))
    assert "pooled validation rows" not in again
    assert "one pooled calibrator" in (tmp_path / "out" / "mock.md").read_text()
