"""Command line: serve, eval, calibrate."""

from __future__ import annotations

import argparse
import json
import sys

from moelars import __version__
from moelars.backends import load_backend
from moelars.calibration import Calibrator
from moelars.engine import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_ROWS, Engine


def _apply_preset(args: argparse.Namespace) -> None:
    """Fill backend, model, adapter and calibration from `--preset`; explicit flags win."""
    from moelars.presets import PRESETS

    name = getattr(args, "preset", None)
    if not name:
        return
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {', '.join(sorted(PRESETS))}")
    preset = PRESETS[name]
    if args.backend in (None, "mock"):
        args.backend = preset.backend
    args.model = args.model or preset.model
    args.adapter = args.adapter or preset.adapter
    args.calibration = args.calibration or preset.calibration


def _engine_from_args(args: argparse.Namespace) -> Engine:
    from moelars.presets import resolve_adapter, resolve_calibration

    _apply_preset(args)
    backend = load_backend(args.backend, model=args.model, template=args.template,
                           adapter=resolve_adapter(args.adapter))
    calibration = resolve_calibration(args.calibration)
    calibrator = Calibrator.load(calibration) if calibration else None
    head = None
    if args.head:
        from moelars.heads import PointerHeadScorer

        head = PointerHeadScorer.load(args.head, args.projection or str(args.head).replace(".npz", ".projection.npy"))
    budgets = {}
    if hasattr(args, "max_rows"):  # serve only; eval and calibrate score one row per question
        budgets = {"max_rows": args.max_rows or None, "max_input_tokens": args.max_input_tokens or None}
    return Engine(backend, calibrator=calibrator, version=__version__, head=head, **budgets)


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default=None, help="named configuration, e.g. 30b (see moelars.presets)")
    parser.add_argument("--backend", default="mock", choices=["mock", "mlx", "llamacpp"])
    parser.add_argument("--model", default=None, help="Model path or Hugging Face id for the backend")
    parser.add_argument("--template", default=None, help="Chat template name: plain, chatml, gemma, llama3")
    parser.add_argument("--calibration", default=None, help="Path to a calibrator JSON produced by `moelars calibrate`")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter directory from `python -m moelars.train.lora`, or a Hugging Face repo ID")
    parser.add_argument("--head", default=None, help="Pointer head npz from `python -m moelars.train.residual`")
    parser.add_argument("--projection", default=None, help="projection.npy from feature extraction")


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from moelars.server import create_app

    engine = _engine_from_args(args)
    app = create_app(engine, max_body_bytes=args.max_body_bytes)
    print(f"moe-LARS {__version__} serving {engine.model_id} on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from moelars.evalset import evaluate, read_examples

    engine = _engine_from_args(args)
    examples = list(read_examples(args.data))
    if args.limit:
        examples = examples[: args.limit]
    result = evaluate(engine, examples)
    print(json.dumps(result.__dict__, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from moelars.evalset import calibrate, read_examples

    engine = _engine_from_args(args)
    examples = list(read_examples(args.data))
    if args.limit:
        examples = examples[: args.limit]
    calibrator = calibrate(engine, examples, source=args.data)
    calibrator.save(args.out)
    print(json.dumps({"temperatures": calibrator.temperatures, "saved": args.out}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moelars", description="Moe Limited but Accurate Response System")
    parser.add_argument("--version", action="version", version=f"moelars {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    _add_backend_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8600)
    serve.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                       help="model rows one request may plan (options x permutations x ablations); 0 for no limit")
    serve.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS,
                       help="input tokens one request may need; 0 for no limit")
    serve.add_argument("--max-body-bytes", type=int, default=1_000_000, help="largest request body accepted")
    serve.set_defaults(func=cmd_serve)

    ev = sub.add_parser("eval", help="Score a labeled JSONL set")
    _add_backend_args(ev)
    ev.add_argument("--data", required=True)
    ev.add_argument("--limit", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    cal = sub.add_parser("calibrate", help="Fit temperatures on a labeled JSONL set")
    _add_backend_args(cal)
    cal.add_argument("--data", required=True)
    cal.add_argument("--out", required=True)
    cal.add_argument("--limit", type=int, default=0)
    cal.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
