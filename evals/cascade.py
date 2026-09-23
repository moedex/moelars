"""Simulate an `escalate_to` cascade offline from two suites' per-row dumps.

Both suites must have been run with `--dump-rows` on the same data and `--rows`, so row i is
the same example in each. A row goes to the fallback model when the primary's calibrated
top probability is below a floor. The floor is one number for every config, chosen on the
validation rows only (best macro accuracy, ties to the lower escalation rate), then applied
to test. The sweep over floors is printed too, so the cost side is visible.

    uv run python evals/cascade.py evals/results/rows/<primary-slug> evals/results/rows/<fallback-slug>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLOORS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


def load(directory: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def combine(primary: list[dict], fallback: list[dict], floor: float) -> tuple[float, float, float]:
    """Accuracy, Brier, and escalation rate when rows under `floor` take the fallback's answer."""
    if len(primary) != len(fallback):
        raise ValueError(f"row counts differ: {len(primary)} vs {len(fallback)}")
    hits, briers, escalated = [], [], 0
    for a, b in zip(primary, fallback, strict=True):
        if a["target"] != b["target"]:
            raise ValueError("rows are not aligned: targets differ")
        row = b if max(a["p"]) < floor else a
        escalated += row is b
        hits.append(row["hit"])
        briers.append(float(np.sum((np.asarray(row["p"]) - np.asarray(row["target"])) ** 2)))
    n = max(len(primary), 1)
    return float(np.mean(hits)), float(np.mean(briers)), escalated / n


def sweep(primary: dict, fallback: dict, split: str, configs: list[str]) -> dict[float, dict]:
    out = {}
    for floor in FLOORS:
        per = {c: combine(primary[c][split], fallback[c][split], floor) for c in configs}
        out[floor] = {
            "acc": float(np.mean([v[0] for v in per.values()])),
            "brier": float(np.mean([v[1] for v in per.values()])),
            "escalated": float(np.mean([v[2] for v in per.values()])),
            "per_config": per,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("fallback", type=Path)
    args = parser.parse_args()
    primary, fallback = load(args.primary), load(args.fallback)
    configs = sorted(set(primary) & set(fallback))
    # A config with no validation rows (chaosnli) is scored on test but cannot vote on the floor.
    val_configs = [c for c in configs if primary[c]["validation"] and fallback[c]["validation"]]
    val = sweep(primary, fallback, "validation", val_configs)
    test = sweep(primary, fallback, "test", configs)
    chosen = max(FLOORS, key=lambda f: (round(val[f]["acc"], 4), -val[f]["escalated"]))

    print(f"{len(configs)} configs; floor chosen on validation ({len(val_configs)} configs): {chosen}\n")
    print("| floor | val acc | test acc | test Brier | escalated (test) |")
    print("|---|---|---|---|---|")
    for f in FLOORS:
        mark = " **chosen**" if f == chosen else ""
        print(f"| {f:.2f}{mark} | {val[f]['acc']:.3f} | {test[f]['acc']:.3f} | {test[f]['brier']:.3f} "
              f"| {test[f]['escalated']:.1%} |")
    only_fallback = test[FLOORS[-1]]
    print(f"\nprimary alone {test[0.0]['acc']:.3f}, fallback alone {only_fallback['acc']:.3f} "
          f"(floor 1.0 escalates all but rows at exactly 1.0: {only_fallback['escalated']:.1%})\n")
    print("| config | primary | fallback | cascade | escalated |")
    print("|---|---|---|---|---|")
    for c in configs:
        a = combine(primary[c]["test"], primary[c]["test"], 0.0)[0]
        b = combine(fallback[c]["test"], fallback[c]["test"], 0.0)[0]
        acc, _, esc = test[chosen]["per_config"][c]
        print(f"| {c} | {a:.3f} | {b:.3f} | {acc:.3f} | {esc:.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
