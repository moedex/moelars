"""Run raw and calibrated evals over jev-bench configs with one loaded backend.

Writes results incrementally so a crash keeps what finished, skips configs already in
the output unless --force, and renders a markdown table with the published Jev column.

    uv run python evals/run_suite.py --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit

With a Tier B head, results and calibrators are written under a separate tag so the
head-plus-per-config-calibration run sits next to the Tier A run for the same model:

    uv run python evals/run_suite.py --backend mlx --model <model> --head checkpoints/pointer_head.npz \
        --projection data/features/projection.npy --tag head
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

from moelars import __version__
from moelars.backends import load_backend
from moelars.calibration import Calibrator
from moelars.engine import Engine
from moelars.evalset import calibrate, evaluate, read_examples

HERE = Path(__file__).resolve().parent

# mnli before chaosnli so chaosnli can borrow mnli's calibrator (it has no validation split).
ALL_CONFIGS = [
    "banking77", "boolq", "sst5",
    "clinc150", "massive", "ledgar", "go_emotions", "mmlu", "arc_challenge", "mnli", "chaosnli",
    "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity", "stsb", "measuring_hate_speech",
    "fever_evidence", "paws", "civil_comments", "sms_spam", "strategyqa_closed", "strategyqa_grounded",
]
CALIBRATION_FALLBACK = {"chaosnli": "mnli"}


def peak_memory_gb(backend) -> float | None:
    """Peak device memory so far for MLX backends; None elsewhere."""
    mx = getattr(backend, "_mx", None)
    if mx is None or not hasattr(mx, "get_peak_memory"):
        return None
    return round(mx.get_peak_memory() / 1e9, 2)


def slugify(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower().split("/")[-1]).strip("-")


def macro(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def render_table(results: dict, jev: dict, path: Path) -> None:
    peak = max((e.get("peak_memory_gb") or 0.0 for e in results["configs"].values()), default=0.0)
    lines = [
        f"# {results['model']}",
        "",
        f"Backend `{results['backend']}`, {results['rows_per_split']} test rows per config, calibration fitted on "
        f"{results['rows_per_split']} validation rows. moe-LARS {results['version']}"
        + (f", pointer head `{results['head']}`" if results.get("head") else "")
        + ". Jev column quoted from jev-bench's published jev-1.13.0 run on full splits."
        + (f" Model load {results['load_s']}s." if results.get("load_s") is not None else "")
        + (f" Peak memory {peak:.1f} GB." if peak else ""),
        "",
        "| config | prim | K | acc raw | acc cal | ECE raw | ECE cal | Brier raw | Brier cal | cov@5% | ms/row "
        "| Jev acc | Jev ECE | Jev Brier |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    summary_rows = []
    for cfg in ALL_CONFIGS:
        entry = results["configs"].get(cfg)
        if not entry:
            continue
        raw, cal, pub = entry["raw"], entry.get("calibrated"), jev.get(cfg, {})
        kind = next(iter(raw["per_kind"]))
        c = cal or raw
        lines.append(
            f"| {cfg} | {kind} | {entry.get('k', '')} | {raw['accuracy']:.3f} | {c['accuracy']:.3f} | "
            f"{raw['ece']:.3f} | {c['ece']:.3f} | {raw['brier']:.3f} | {c['brier']:.3f} | "
            f"{c['coverage_at_5pct']:.2f} | {raw['ms_per_example']:.0f} | "
            f"{pub.get('acc', '')} | {pub.get('ece', '')} | {pub.get('brier', '')} |"
        )
        summary_rows.append(
            {"kind": kind, "acc": c["accuracy"], "ece": c["ece"], "brier": c["brier"],
             "jev_acc": pub.get("acc"), "jev_ece": pub.get("ece"), "jev_brier": pub.get("brier")}
        )
    lines += ["", "## Macro averages over the configs above (calibrated)", "",
              "| scope | n | acc | ECE | Brier | Jev acc | Jev ECE | Jev Brier |", "|---|---|---|---|---|---|---|---|"]
    for scope in ["all", "choice", "score", "noul"]:
        rows = summary_rows if scope == "all" else [r for r in summary_rows if r["kind"] == scope]
        if not rows:
            continue
        lines.append(
            f"| {scope} | {len(rows)} | {macro(rows, 'acc')} | {macro(rows, 'ece')} | {macro(rows, 'brier')} | "
            f"{macro(rows, 'jev_acc')} | {macro(rows, 'jev_ece')} | {macro(rows, 'jev_brier')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default=None)
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    parser.add_argument("--calibration-dir", default=str(HERE.parent / "calibration"))
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory from moelars.train.lora")
    parser.add_argument("--head", default=None, help="Tier B pointer head npz; runs every pass through it")
    parser.add_argument("--projection", default=None, help="projection.npy that the head was trained with")
    parser.add_argument("--tag", default=None, help="suffix for the result and calibrator files, e.g. 'head'")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dump-rows", action="store_true",
                        help="also write calibrated per-row probabilities for test and validation to "
                             "<out-dir>/rows/<slug>/<config>.json (costs one more validation pass)")
    args = parser.parse_args()

    slug = slugify(args.model) + (f"-{args.tag}" if args.tag else "")
    out_json = Path(args.out_dir) / f"{slug}.json"
    out_md = Path(args.out_dir) / f"{slug}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    Path(args.calibration_dir).mkdir(parents=True, exist_ok=True)
    jev = json.loads((HERE / "jev_published.json").read_text())

    results = json.loads(out_json.read_text()) if out_json.exists() and not args.force else {}
    results.update({"model": args.model, "backend": args.backend, "rows_per_split": args.rows, "version": __version__,
                    "adapter": args.adapter, "head": args.head})
    results.setdefault("configs", {})

    load_started = time.perf_counter()
    backend = load_backend(args.backend, model=args.model, template=args.template, adapter=args.adapter)
    results["load_s"] = round(time.perf_counter() - load_started, 1)
    head = None
    if args.head:
        from moelars.heads import PointerHeadScorer

        head = PointerHeadScorer.load(args.head, args.projection or args.head.replace(".npz", ".projection.npy"))
    calibrators: dict[str, Calibrator] = {}

    def engine(calibrator: Calibrator | None = None) -> Engine:
        return Engine(backend, calibrator=calibrator, head=head)

    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if cfg in results["configs"] and not args.force:
            print(f"[{cfg}] cached", flush=True)
            continue
        test_path = Path(args.data_dir) / f"{cfg}.test.jsonl"
        val_path = Path(args.data_dir) / f"{cfg}.validation.jsonl"
        if not test_path.exists():
            print(f"[{cfg}] missing {test_path}", flush=True)
            continue
        test = list(read_examples(test_path))[: args.rows]
        started = time.perf_counter()
        raw = evaluate(engine(), test)

        calibrator: Calibrator | None = None
        if val_path.exists() and val_path.stat().st_size > 0:
            val = list(read_examples(val_path))[: args.rows]
            calibrator = calibrate(engine(), val, source=f"jev-bench/{cfg}/validation[:{len(val)}]")
        elif cfg in CALIBRATION_FALLBACK:
            fallback = CALIBRATION_FALLBACK[cfg]
            fallback_path = Path(args.calibration_dir) / f"{fallback}.{slug}.json"
            if fallback in calibrators:
                calibrator = calibrators[fallback]
            elif fallback_path.exists():
                calibrator = Calibrator.load(fallback_path)
        if calibrator is not None:
            calibrators[cfg] = calibrator
            calibrator.save(Path(args.calibration_dir) / f"{cfg}.{slug}.json")
        test_rows: list[dict] | None = [] if args.dump_rows else None
        calibrated = evaluate(engine(calibrator), test, rows=test_rows) if calibrator else None
        if args.dump_rows:
            if calibrated is None:
                evaluate(engine(), test, rows=test_rows)
            val_rows: list[dict] = []
            if val_path.exists() and val_path.stat().st_size > 0:
                evaluate(engine(calibrator), list(read_examples(val_path))[: args.rows], rows=val_rows)
            rows_path = Path(args.out_dir) / "rows" / slug / f"{cfg}.json"
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            rows_path.write_text(json.dumps({"test": test_rows, "validation": val_rows}))

        first_question = test[0].question
        k = len(first_question.get("criteria") or []) if first_question["type"] != "noul" else 2
        entry = {"k": k, "raw": asdict(raw), "calibrated": asdict(calibrated) if calibrated else None,
                 "elapsed_s": round(time.perf_counter() - started, 1), "peak_memory_gb": peak_memory_gb(backend)}
        results["configs"][cfg] = entry
        out_json.write_text(json.dumps(results, indent=2))
        render_table(results, jev, out_md)
        c = calibrated or raw
        print(
            f"[{cfg}] raw acc={raw.accuracy:.3f} ece={raw.ece:.3f} | cal acc={c.accuracy:.3f} ece={c.ece:.3f} "
            f"brier={c.brier:.3f} cov@5%={c.coverage_at_5pct:.2f} | {raw.ms_per_example:.0f}ms/row "
            f"{entry['elapsed_s']}s",
            flush=True,
        )
    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
