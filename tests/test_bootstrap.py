"""evals/bootstrap.py pairs rows across systems and scores `+` blends like evals/cascade.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from bootstrap import outcomes  # noqa: E402


def _dump(directory: Path, rows: list[tuple[list[float], list[float], bool]]) -> Path:
    directory.mkdir()
    test = [{"id": str(i), "p": p, "target": t, "hit": h} for i, (p, t, h) in enumerate(rows)]
    (directory / "cfg.json").write_text(json.dumps({"test": test, "validation": []}))
    return directory


def test_blend_averages_probabilities_and_scores_the_target_top(tmp_path):
    a = _dump(tmp_path / "a", [([0.6, 0.4], [1, 0], True), ([0.45, 0.55], [1, 0], False)])
    b = _dump(tmp_path / "b", [([0.2, 0.8], [1, 0], False), ([0.9, 0.1], [1, 0], True)])
    hits, briers = outcomes(f"{a}+{b}")["cfg"]
    assert hits.tolist() == [0.0, 1.0]  # (0.4, 0.6) misses; (0.675, 0.325) hits
    assert briers[1] == pytest.approx(2 * 0.325**2)
    assert np.array_equal(outcomes(str(a))["cfg"][0], [1.0, 0.0])


def test_misaligned_rows_are_refused(tmp_path):
    a = _dump(tmp_path / "a", [([0.6, 0.4], [1, 0], True)])
    b = _dump(tmp_path / "b", [([0.6, 0.4], [1, 0], True)])
    rows = json.loads((b / "cfg.json").read_text())
    rows["test"][0]["id"] = "other"
    (b / "cfg.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="not aligned"):
        outcomes(f"{a}+{b}")
