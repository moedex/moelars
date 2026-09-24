import json

import numpy as np
import pytest

from moelars.backends.mock import MockBackend
from moelars.calibration import Calibrator
from moelars.engine import Engine
from moelars.evalset import calibrate, evaluate, read_examples
from moelars.schema import SystemOneRequest


def _write(tmp_path, rows):
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_reads_jev_bench_style_encoded_fields(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "state": json.dumps("Is there a virtual card option?"),
                "question": json.dumps({"type": "noul", "instructions": "Asks about a card"}),
                "label": "1",
                "soft_label": json.dumps({"yes": 0.8, "no": 0.2}),
            }
        ],
    )
    example = next(read_examples(path))
    assert example.state == "Is there a virtual card option?"
    assert example.question["type"] == "noul"
    assert example.soft_label == {"yes": 0.8, "no": 0.2}


def test_calibrate_fits_platt_for_nouls_and_temperature_for_choice(tmp_path):
    noul = {"type": "noul", "instructions": "Mentions payouts"}
    choice = {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "technical": None}}
    rows = []
    for i in range(12):
        rows.append({"state": f"message {i} about payouts", "question": noul, "label": "1"})
        rows.append({"state": f"message {i} about weather", "question": noul, "label": "0"})
        rows.append({"state": f"ticket {i}: billing problem", "question": choice, "label": "billing"})
    engine = Engine(MockBackend())
    calibrator = calibrate(engine, list(read_examples(_write(tmp_path, rows))), source="unit")
    assert "noul" in calibrator.platt
    assert calibrator.temperatures["noul"] == 1.0
    assert calibrator.temperatures["choice"] > 0
    result = evaluate(Engine(MockBackend(), calibrator=calibrator), list(read_examples(_write(tmp_path, rows))))
    assert result.count == 36
    assert result.platt is not None and result.platt["noul"] is not None
    assert 0 <= result.accuracy <= 1


def test_platt_bias_moves_noul_across_half():
    question = {"type": "noul", "instructions": "Urgent?"}
    request = SystemOneRequest(state="Thanks, that fixed it!", questions={"q": question})
    neutral = Engine(MockBackend()).evaluate(request).answers["q"].noul
    biased = Calibrator(platt={"noul": (1.0, 12.0)})
    pushed = Engine(MockBackend(), calibrator=biased).evaluate(request).answers["q"].noul
    assert pushed > neutral
    assert pushed > 0.99


def test_temperature_is_platt_special_case():
    engine_t = Engine(MockBackend(), calibrator=Calibrator(temperatures={"noul": 4.0}))
    engine_p = Engine(MockBackend(), calibrator=Calibrator(platt={"noul": (0.25, 0.0)}))
    for z in (-6.0, -1.0, 0.0, 2.5, 9.0):
        assert np.isclose(engine_t.noul_prob(z), engine_p.noul_prob(z))


def test_row_dumps_carry_a_stable_example_id(tmp_path):
    noul = {"type": "noul", "instructions": "Is this about money?"}
    data = [{"state": "payout failed", "question": noul, "label": "1"},
            {"state": "nice weather", "question": noul, "label": "0"}]
    examples = list(read_examples(_write(tmp_path, data)))
    rows: list[dict] = []
    evaluate(Engine(MockBackend()), examples, rows=rows)
    assert [r["id"] for r in rows] == [e.id for e in examples]
    assert rows[0]["id"] != rows[1]["id"]
    assert [e.id for e in read_examples(_write(tmp_path, data))] == [e.id for e in examples]


def test_cascade_refuses_rows_that_are_different_examples_with_the_same_target():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("cascade", Path(__file__).parents[1] / "evals" / "cascade.py")
    cascade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cascade)
    row = {"p": [0.9, 0.1], "target": [1.0, 0.0], "hit": True}
    assert cascade.combine([{**row, "id": "a"}], [{**row, "id": "a"}], 0.5, "blend")[0] == 1.0
    assert cascade.combine([row], [row], 0.5, "blend")[0] == 1.0  # dumps from before IDs still combine
    with pytest.raises(ValueError, match="not aligned"):
        cascade.combine([{**row, "id": "a"}], [{**row, "id": "b"}], 0.5, "blend")
