import numpy as np

from moelar.calibration import Calibrator, brier, coverage_at_error, ece, fit_platt, fit_temperature


def test_fit_temperature_recovers_flatter_scale_for_overconfident_logits():
    rng = np.random.default_rng(0)
    logits, targets = [], []
    for _ in range(200):
        true = rng.integers(0, 3)
        row = rng.normal(0, 1, 3) * 4  # overconfident
        row[true] += 2.0
        if rng.random() < 0.3:  # 30% label noise makes sharp logits wrong
            true = rng.integers(0, 3)
        logits.append(row)
        targets.append(np.eye(3)[true])
    t = fit_temperature(logits, targets)
    assert t > 1.0


def test_fit_platt_learns_bias():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 2, 500)
    labels = (scores + 1.5 > 0).astype(float)  # shifted decision boundary
    a, b = fit_platt(scores, labels)
    assert a > 0
    assert b > 0.5


def test_ece_and_brier_basic():
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    hit = np.array([1, 1, 0, 1])
    assert 0 <= ece(conf, hit) <= 1
    assert brier([np.array([1.0, 0.0])], [np.array([1.0, 0.0])]) == 0.0


def test_coverage_at_error():
    conf = np.array([0.99, 0.95, 0.7, 0.6])
    hit = np.array([1, 1, 0, 1])
    cov, thr = coverage_at_error(conf, hit, 0.0)
    assert cov == 0.5 and thr == 0.95


def test_calibrator_roundtrip(tmp_path):
    c = Calibrator(temperatures={"noul": 1.7}, platt={"noul": (0.9, 0.2)}, fitted_on="x")
    path = tmp_path / "cal.json"
    c.save(path)
    loaded = Calibrator.load(path)
    assert loaded.temperature_for("noul") == 1.7
    assert loaded.platt_for("noul") == (0.9, 0.2)
    assert loaded.temperature_for("choice") == 1.0
