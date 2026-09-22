# MoeLAR

**Moe Limited but Accurate Response.** A local-models-only typed-decision engine.

State plus typed questions go in. Calibrated probability distributions come out, in one
forward pass per question, with no text generation. The answer space is limited to the
options you declare, which is what makes calibration and zero structural errors possible.

MoeLAR is wire-compatible with the System One HTTP API. Point any existing System One
client at it by changing the base URL.

```
POST /v1/systemone
GET  /v1/models
```

Three primitives from the System One shape, one MoeLAR addition:

| primitive | question | answer |
|---|---|---|
| `noul` | is this true? | `noul`: P(yes) |
| `choice` | which one of these? | `choice`, `probabilities`, `confidence` |
| `score` | which ordered level? | `score` (may be fractional), `legend`, `probabilities`, `confidence` |
| `multi` | which of these apply? | `probabilities` per option, `selected` |

MoeLAR extensions, all opt-in under a `moelar` request key: permutation-averaged choice
answers with an `order_sensitivity` metric, declared constraints between nouls,
abstention below a probability margin, and evidence spans by leave-one-out ablation.

## Quickstart

```bash
uv sync --extra dev
uv run moelar serve --backend mock            # no model needed, deterministic demo answers
```

```bash
curl -s localhost:8600/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, payouts", "technical": "Bugs, outages"}},
    "is_urgent":  {"type": "noul", "instructions": "The message conveys urgency"}
  }
}'
```

With a real model on Apple Silicon:

```bash
uv sync --extra dev --extra mlx
uv run moelar serve --backend mlx --model mlx-community/Qwen3.5-4B-Instruct-4bit
```

Or any GGUF model anywhere:

```bash
uv sync --extra dev --extra llamacpp
uv run moelar serve --backend llamacpp --model ./models/qwen3.5-4b-instruct-q4_k_m.gguf --template chatml
```

Use it from the official System One SDKs by swapping the base URL:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8600")
r = client.system_one(
    state="I was charged twice.",
    questions={"refund": Noul("Is the customer asking for money back?"),
               "team": Choice("Which team?", criteria={"billing": None, "technical": None})},
)
print(r.nouls["refund"].noul, r.choices["team"].choice)
```

## Calibrate on your own data

Raw model logits are not calibrated. Fit temperatures on a labeled set, then serve with them:

```bash
uv run moelar calibrate --backend mlx --model <model> --data my_labels.jsonl --out calibration/mine.json
uv run moelar serve     --backend mlx --model <model> --calibration calibration/mine.json
uv run moelar eval      --backend mlx --model <model> --calibration calibration/mine.json --data my_test.jsonl
```

`eval` reports accuracy, expected calibration error, Brier score, and the coverage you
can automate at a 5% error budget. The JSONL format is in `evals/README.md`. Nouls get
a Platt fit on the raw yes-minus-no logit, which can move a biased model's decision
boundary; choice and score get a temperature.

## Train a decision head (Tier B)

The backbone stays frozen. A small zero-initialized pointer head learns a residual on
the model's own label logits from cached features, in seconds, and serves through the
same engine:

```bash
uv run python -m moelar.train.build   --out data/train
uv run python -m moelar.train.extract --model <model> --records data/train/*.train.jsonl --out data/features
uv run python -m moelar.train.residual --train data/features/train.npz --heldout data/features/heldout.npz
uv run moelar serve --backend mlx --model <model> --head checkpoints/pointer_head.npz --projection data/features/projection.npy
```

Details and the data policy are in `src/moelar/train/README.md`.

## Layout

```
src/moelar/
  schema.py        wire models, request validation, MoeLAR extensions
  render.py        prompt rows: fenced state prefix + per-question suffix
  labels.py        single-token option labels verified per tokenizer
  primitives.py    softmax, confidence formulas, expected score, order sensitivity
  calibration.py   temperature and Platt fitting, ECE, Brier, coverage-at-error
  engine.py        rows -> backend -> answers; permutations, constraints, abstain, evidence
  backends/        mock, mlx, llamacpp
  server.py        FastAPI app, System One error and header conventions
  evalset.py       labeled JSONL loading, eval and calibrate
  cli.py           moelar serve | eval | calibrate
tests/             unit tests plus a conformance test that runs the official SDK in-process
evals/             benchmark data conventions and a jev-bench fetcher
```

## Status

Pre-alpha. The mock backend and the HTTP contract are tested. The MLX backend has been
run on Apple Silicon with Qwen3-4B-Instruct and benchmarked on three jev-bench configs;
see `evals/RESULTS.md`. The llama.cpp backend is written against its library's documented
API and still needs a hardware pass. See `DESIGN.md` for the architecture, the reasoning,
and the roadmap.

## License

Apache-2.0.
