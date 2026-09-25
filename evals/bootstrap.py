"""Paired bootstrap over test rows: is one system's macro accuracy really above another's?

Each system is a row-dump directory from `run_suite.py --dump-rows`, or several joined by
`+`, which averages their probabilities per row (the `blend` of `evals/cascade.py`, and
what `EnsembleEngine` serves). The first system is the baseline; every other one is
compared with it. Rows are resampled within each config, with the same draw for every
system, so the interval reflects test-row noise on paired answers. It says nothing about
seed variance: compare two seeds' dumps for that.

    uv run python evals/bootstrap.py evals/results/rows/<a> evals/results/rows/<a>+evals/results/rows/<b>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import load  # noqa: E402


def outcomes(spec: str, configs: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per config, each test row's hit and Brier for one system (a dump directory, or a `+` blend)."""
    dumps = [load(Path(part)) for part in spec.split("+")]
    shared = sorted(set.intersection(*(set(d) for d in dumps)))
    result = {}
    for cfg in configs or shared:
        rows = [d[cfg]["test"] for d in dumps]
        if len({len(r) for r in rows}) != 1:
            raise ValueError(f"[{cfg}] row counts differ across {spec}")
        hits, briers = [], []
        for aligned in zip(*rows, strict=True):
            if len({row.get("id") for row in aligned}) != 1:
                raise ValueError(f"[{cfg}] rows are not aligned in {spec}")
            target = np.asarray(aligned[0]["target"])
            if len(aligned) == 1:
                p, hit = np.asarray(aligned[0]["p"]), aligned[0]["hit"]
            else:
                p = np.mean([np.asarray(row["p"]) for row in aligned], axis=0)
                hit = bool(target[int(np.argmax(p))] == target.max())
            hits.append(float(hit))
            briers.append(float(np.sum((p - target) ** 2)))
        result[cfg] = (np.asarray(hits), np.asarray(briers))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="+", help="baseline first; a dump directory, or several joined by '+'")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.systems) < 2:
        raise SystemExit("give a baseline and at least one system to compare with it")

    first = [outcomes(s) for s in args.systems]
    configs = sorted(set.intersection(*(set(o) for o in first)))
    systems = [{c: o[c] for c in configs} for o in first]
    rng = np.random.default_rng(args.seed)
    draws = {c: rng.integers(0, len(systems[0][c][0]), size=(args.resamples, len(systems[0][c][0])))
             for c in configs}

    def macro_draws(system: dict, which: int) -> np.ndarray:
        return np.mean([system[c][which][draws[c]].mean(axis=1) for c in configs], axis=0)

    print(f"{len(configs)} configs, {args.resamples} paired resamples of test rows\n")
    print("| system | macro acc | Brier | acc vs baseline (95% CI) |")
    print("|---|---|---|---|")
    base = macro_draws(systems[0], 0)
    for spec, system in zip(args.systems, systems, strict=True):
        acc = float(np.mean([system[c][0].mean() for c in configs]))
        brier = float(np.mean([system[c][1].mean() for c in configs]))
        if system is systems[0]:
            versus = "baseline"
        else:
            diff = macro_draws(system, 0) - base
            point = acc - float(np.mean([systems[0][c][0].mean() for c in configs]))
            low, high = np.percentile(diff, [2.5, 97.5])
            versus = f"{point * 100:+.1f} pts ({low * 100:+.1f} to {high * 100:+.1f})"
        name = " + ".join(Path(part).name for part in spec.split("+"))
        print(f"| {name} | {acc:.3f} | {brier:.3f} | {versus} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
