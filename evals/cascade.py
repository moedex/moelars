"""Simulate an `escalate_to` cascade offline from two suites' per-row dumps.

Both suites must have been run with `--dump-rows` on the same data and `--rows`, so row i
is the same example in each. Rows carry an example ID (a content hash) that must match;
dumps written before IDs existed are checked by target only, with a warning.

A row is escalated when the primary's calibrated top probability is below a floor.
Policies differ in what an escalated row answers with:

- `switch`: the fallback's probabilities.
- `blend`: the mean of both models' probabilities.

and in whether one floor serves every config (`global`) or each primitive gets its own
(`per-kind`, needs `--suite`, the primary's suite JSON, to know each config's primitive).
Every floor is chosen on the validation rows only (best macro accuracy, ties to the lower
escalation rate) and then applied to test. `always blend` (both models on every row) is
printed as the reference for what escalation leaves on the table.

    uv run python evals/cascade.py evals/results/rows/<primary-slug> evals/results/rows/<fallback-slug> \
        --suite evals/results/<primary-slug>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLOORS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
ALWAYS = 1.01  # a floor above any probability escalates every row


def load(directory: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def combine(primary: list[dict], fallback: list[dict], floor: float, mode: str) -> tuple[float, float, float]:
    """Accuracy, Brier, and escalation rate for one config under one policy.

    A row answered by one model keeps that model's recorded hit; a blended row is right when
    it picks the target's top option."""
    if len(primary) != len(fallback):
        raise ValueError(f"row counts differ: {len(primary)} vs {len(fallback)}")
    hits, briers, escalated = [], [], 0
    for a, b in zip(primary, fallback, strict=True):
        if "id" in a and "id" in b and a["id"] != b["id"]:
            raise ValueError(f"rows are not aligned: example {a['id']} against {b['id']}")
        if a["target"] != b["target"]:
            raise ValueError("rows are not aligned: targets differ")
        target, pa, pb = np.asarray(a["target"]), np.asarray(a["p"]), np.asarray(b["p"])
        if pa.max() >= floor:
            p, hit = pa, a["hit"]
        elif mode == "switch":
            p, hit = pb, b["hit"]
        else:
            p = (pa + pb) / 2
            hit = bool(target[int(np.argmax(p))] == target.max())
        escalated += pa.max() < floor
        hits.append(hit)
        briers.append(float(np.sum((p - target) ** 2)))
    n = max(len(primary), 1)
    return float(np.mean(hits)), float(np.mean(briers)), escalated / n


def macro(primary: dict, fallback: dict, split: str, floors: dict[str, float], mode: str) -> dict:
    per = {c: combine(primary[c][split], fallback[c][split], f, mode) for c, f in floors.items()}
    return {
        "acc": float(np.mean([v[0] for v in per.values()])) if per else float("nan"),
        "brier": float(np.mean([v[1] for v in per.values()])) if per else float("nan"),
        "escalated": float(np.mean([v[2] for v in per.values()])) if per else float("nan"),
        "per_config": per,
    }


def has_validation(primary: dict, fallback: dict, config: str) -> bool:
    return bool(primary[config]["validation"] and fallback[config]["validation"])


def choose(primary: dict, fallback: dict, configs: list[str], mode: str) -> float:
    """The floor with the best validation macro accuracy over `configs`, ties to less escalation."""
    usable = [c for c in configs if has_validation(primary, fallback, c)]
    if not usable:
        return 0.0
    scored = {f: macro(primary, fallback, "validation", dict.fromkeys(usable, f), mode) for f in FLOORS}
    return max(FLOORS, key=lambda f: (round(scored[f]["acc"], 4), -scored[f]["escalated"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("fallback", type=Path)
    parser.add_argument("--suite", type=Path, help="primary's suite JSON, for each config's primitive")
    args = parser.parse_args()
    primary, fallback = load(args.primary), load(args.fallback)
    configs = sorted(set(primary) & set(fallback))
    unchecked = [c for c in configs for side in (primary, fallback)
                 if any("id" not in row for split in side[c].values() for row in split)]
    if unchecked:
        print(f"warning: rows without example IDs in {sorted(set(unchecked))}; aligned by target only",
              file=sys.stderr)
    kinds: dict[str, str] = {}
    if args.suite:
        suite = json.loads(args.suite.read_text())["configs"]
        kinds = {c: next(iter(suite[c]["raw"]["per_kind"])) for c in configs if c in suite}

    references = {
        "primary alone": ("switch", dict.fromkeys(configs, 0.0)),
        "fallback alone": ("switch", dict.fromkeys(configs, ALWAYS)),
        "always blend (both models on every row)": ("blend", dict.fromkeys(configs, ALWAYS)),
    }
    policies: dict[str, tuple[str, dict[str, float]]] = dict(references)
    for mode in ("switch", "blend"):
        floor = choose(primary, fallback, configs, mode)
        policies[f"{mode}, global floor {floor}"] = (mode, dict.fromkeys(configs, floor))
        if kinds:
            floors, labels = {}, []
            for kind in sorted(set(kinds.values())):
                members = [c for c in configs if kinds.get(c) == kind]
                f = choose(primary, fallback, members, mode)
                floors.update(dict.fromkeys(members, f))
                labels.append(f"{kind} {f}")
            policies[f"{mode}, per-kind floors ({', '.join(labels)})"] = (mode, floors)

    print(f"{len(configs)} configs; floors chosen on validation rows only\n")
    print("| policy | val acc | test acc | test Brier | escalated (test) |")
    print("|---|---|---|---|---|")
    results = {}
    for name, (mode, floors) in policies.items():
        val_floors = {c: f for c, f in floors.items() if has_validation(primary, fallback, c)}
        val = macro(primary, fallback, "validation", val_floors, mode)
        test = macro(primary, fallback, "test", floors, mode)
        results[name] = test
        print(f"| {name} | {val['acc']:.3f} | {test['acc']:.3f} | {test['brier']:.3f} | {test['escalated']:.1%} |")

    # The per-config view is for reading, not selection: it shows the policy validation chose.
    chosen = max((n for n in policies if n not in references),
                 key=lambda n: macro(primary, fallback, "validation",
                                     {c: f for c, f in policies[n][1].items() if has_validation(primary, fallback, c)},
                                     policies[n][0])["acc"])
    print(f"\nper config, under the policy with the best validation accuracy ({chosen}):\n")
    print("| config | kind | primary | fallback | policy | escalated |")
    print("|---|---|---|---|---|---|")
    for c in configs:
        a = results["primary alone"]["per_config"][c][0]
        b = results["fallback alone"]["per_config"][c][0]
        acc, _, esc = results[chosen]["per_config"][c]
        print(f"| {c} | {kinds.get(c, '')} | {a:.3f} | {b:.3f} | {acc:.3f} | {esc:.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
