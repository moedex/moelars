"""Side-by-side table of several suite result files, with the published Jev column.

    uv run python evals/compare.py evals/results/qwen3-4b-instruct-2507-4bit.json:4B \
        evals/results/qwen3-4b-instruct-2507-4bit-head.json:4B+head --metric acc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_suite import ALL_CONFIGS, macro  # noqa: E402

METRIC_KEY = {"acc": "accuracy", "ece": "ece", "brier": "brier", "cov": "coverage_at_5pct", "ms": "ms_per_example"}


def load(spec: str) -> tuple[str, dict]:
    path, _, name = spec.partition(":")
    data = json.loads(Path(path).read_text())
    return name or Path(path).stem, data["configs"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="path[:column name]")
    parser.add_argument("--metric", default="acc", choices=sorted(METRIC_KEY))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    columns = [load(s) for s in args.results]
    jev = json.loads((HERE / "jev_published.json").read_text())
    key = METRIC_KEY[args.metric]
    lines = ["| config | prim | K | " + " | ".join(name for name, _ in columns) + " | Jev |",
             "|---|---|---|" + "---|" * (len(columns) + 1)]
    rows = []
    for cfg in ALL_CONFIGS:
        entries = [c.get(cfg) for _, c in columns]
        if not any(entries):
            continue
        first = next(e for e in entries if e)
        kind = next(iter(first["raw"]["per_kind"]))
        values = [((e["raw"] if args.metric == "ms" else (e["calibrated"] or e["raw"]))[key] if e else None)
                  for e in entries]
        pub = jev.get(cfg, {}).get(args.metric)
        best = None
        if args.metric in {"acc", "cov"}:
            best = max(v for v in values if v is not None)
        elif any(v is not None for v in values):
            best = min(v for v in values if v is not None)
        reported = sum(x is not None for x in values)
        cells = [(f"**{v:.3f}**" if v == best and reported > 1 else f"{v:.3f}") if v is not None else ""
                 for v in values]
        lines.append(f"| {cfg} | {kind} | {first['k']} | " + " | ".join(cells)
                     + f" | {'' if pub is None else f'{pub:.3f}'} |")
        rows.append({"kind": kind, **{name: v for (name, _), v in zip(columns, values, strict=True)}, "jev": pub})
    lines += ["", "| macro | n | " + " | ".join(name for name, _ in columns) + " | Jev |",
              "|---|---|" + "---|" * (len(columns) + 1)]
    for scope in ["all", "choice", "score", "noul"]:
        subset = rows if scope == "all" else [r for r in rows if r["kind"] == scope]
        if not subset:
            continue
        complete = [r for r in subset if all(r[name] is not None for name, _ in columns)]
        lines.append(f"| {scope} | {len(complete)} | "
                     + " | ".join(f"{macro(complete, name)}" for name, _ in columns)
                     + f" | {macro(complete, 'jev')} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
