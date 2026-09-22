"""Calibration.

Raw label logits from a language model are not calibrated. MoeLAR fits, on the
user's own labeled data:

- a temperature per primitive kind (noul, choice, score, multi), which reshapes
  distributions without changing the argmax
- an optional Platt scaler (a, b) for noul-style probabilities, which can also move a
  decision across 0.5 when the raw model is biased toward one side

Metrics: expected calibration error (ECE), Brier score, and coverage at an error budget.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from moelar.primitives import softmax

KINDS = ("noul", "choice", "score", "multi")


@dataclass
class Calibrator:
    temperatures: dict[str, float] = field(default_factory=lambda: dict.fromkeys(KINDS, 1.0))
    platt: dict[str, tuple[float, float]] = field(default_factory=dict)
    fitted_on: str | None = None

    def temperature_for(self, kind: str) -> float:
        return float(self.temperatures.get(kind, 1.0))

    def platt_for(self, kind: str) -> tuple[float, float] | None:
        value = self.platt.get(kind)
        return (float(value[0]), float(value[1])) if value else None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        raw = json.loads(Path(path).read_text())
        platt = {k: (float(v[0]), float(v[1])) for k, v in raw.get("platt", {}).items()}
        return cls(temperatures=raw.get("temperatures", {}), platt=platt, fitted_on=raw.get("fitted_on"))


# --------------------------------------------------------------------------- fitting


def fit_temperature(logit_rows: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Grid-search the temperature minimizing cross-entropy against (possibly soft) targets."""
    if not logit_rows:
        return 1.0
    grid = np.geomspace(0.05, 20.0, 400)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for logits, target in zip(logit_rows, targets, strict=True):
            p = np.clip(softmax(logits, t), 1e-9, 1.0)
            nll -= float((np.asarray(target) * np.log(p)).sum())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def _logistic_loss(z: np.ndarray, y: np.ndarray) -> float:
    # log(1 + exp(z)) - y * z, computed stably
    return float((np.logaddexp(0.0, z) - y * z).sum())


def fit_platt(scores: np.ndarray, labels: np.ndarray, l2: float = 1e-2, iterations: int = 100) -> tuple[float, float]:
    """Fit sigmoid(a * s + b) to binary labels.

    Scores are standardized before fitting so the L2 penalty and the step sizes are
    scale-free, and each Newton step is backtracked until the penalized loss decreases.
    Raw language-model logit differences can be tens of units wide and nearly separable,
    which makes an undamped Newton fit overshoot into a flipped or exploding solution.
    Platt's original label smoothing keeps the separable case finite.
    """
    s_raw = np.asarray(scores, dtype=np.float64)
    y_raw = np.asarray(labels, dtype=np.float64)
    if s_raw.size == 0:
        return 1.0, 0.0
    scale = float(s_raw.std()) or 1.0
    s = s_raw / scale
    n_pos, n_neg = float(y_raw.sum()), float((1 - y_raw).sum())
    y = np.where(y_raw > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)) if n_pos and n_neg else y_raw

    def penalized(a: float, b: float) -> float:
        return _logistic_loss(a * s + b, y) + 0.5 * l2 * a * a

    a, b = 1.0, 0.0
    current = penalized(a, b)
    for _ in range(iterations):
        z = a * s + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        w = p * (1 - p) + 1e-12
        grad_a = float(((p - y) * s).sum() + l2 * a)
        grad_b = float((p - y).sum())
        h_aa = float((w * s * s).sum() + l2)
        h_ab = float((w * s).sum())
        h_bb = float(w.sum())
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * grad_a - h_ab * grad_b) / det
        db = (h_aa * grad_b - h_ab * grad_a) / det
        step = 1.0
        improved = False
        for _ in range(30):
            candidate = penalized(a - step * da, b - step * db)
            if candidate < current:
                a, b, current, improved = a - step * da, b - step * db, candidate, True
                break
            step *= 0.5
        if not improved or (abs(step * da) < 1e-9 and abs(step * db) < 1e-9):
            break
    return float(a / scale), float(b)


# --------------------------------------------------------------------------- metrics


def ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def brier(probs: list[np.ndarray], targets: list[np.ndarray]) -> float:
    if not probs:
        return 0.0
    return float(np.mean([((np.asarray(p) - np.asarray(t)) ** 2).sum() for p, t in zip(probs, targets, strict=True)]))


def coverage_at_error(confidences: np.ndarray, correct: np.ndarray, max_error: float = 0.05) -> tuple[float, float]:
    """Largest fraction of decisions acceptable above one threshold with empirical error <= max_error.

    Returns (coverage, threshold). Coverage is 0 when no threshold meets the budget.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0, 1.0
    order = np.argsort(-conf)
    best_cov, best_thr = 0.0, 1.0
    for k in range(1, conf.size + 1):
        taken = order[:k]
        if 1.0 - hit[taken].mean() <= max_error:
            best_cov, best_thr = k / conf.size, float(conf[order[k - 1]])
    return best_cov, best_thr
