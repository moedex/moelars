# Handoff, 2026-09-22 evening

Written before a reboot. Everything below is committed and pushed to `origin/main`
(`git@github.com:moedex/moelar.git`). Nothing is running.

## State in one paragraph

MoeLAR serves the System One wire shape from a local model with no text generation.
Tier A (frozen Qwen3-4B-Instruct, per-config calibration) sits at 0.662 macro accuracy
on jev-bench against Jev's published 0.733, with better calibration (ECE 0.088 vs
0.113). A Tier B pointer head, trained in seconds on cached features, was fairly
compared today: +1.8 macro accuracy with per-config calibration on both sides, all of it
on question forms the head trained on, neutral on held-out forms, no latency cost.
Full tables: `evals/RESULTS.md`, `evals/results/compare-qwen3-4b-head.md`.

## What happened in this batch

- `evals/run_suite.py` takes `--head/--projection/--tag` and records model load time and
  peak memory per run. `evals/compare.py` renders several runs side by side for
  `acc|ece|brier|ms` with the Jev column.
- The MLX backend reaches the text stack through `language_model` when present, and
  passes `enable_thinking=False` to chat templates, both for Qwen3.5. Unit tests in
  `tests/test_mlx_backend.py`; the 9B model has **not** been run yet.
- `examples/molar_triage/`: hand-written dental inbox (60 messages, noul + choice +
  score) as the calibrate-on-your-own-data walkthrough. Files, request sample, README,
  and a test. Its "Numbers" section is a placeholder until the queue below runs.
- Suite results for the head run and its 22 calibrators are committed under
  `evals/results/*-head.*` and `calibration/*-head.json`.

## What was queued and cancelled (run this next)

`scripts/next_batch.sh` is the exact queue, serial, four to five hours on the M5 Max:

1. Extract features for the full corpus (about 14,000 records, `--limit 20000`) with
   Qwen3-4B, train head v2, per-source report, then the fair suite with `--tag head-v2`.
2. `scripts/smoke_mlx_backend.py` on `mlx-community/Qwen3.5-9B-MLX-4bit` (already in
   the HF cache, 1.3 GB fetched). It checks the prefix-cache restore on the hybrid
   Gated DeltaNet + attention cache, which the backend snapshots via `.state`; if the
   check fails, that is the first thing to fix. Then Tier A suite, extraction, head,
   and head suite on 9B.
3. Molar Triage numbers for both models with and without heads, and
   `scripts/load_cost.py` for load time, peak memory, and warm request latency.

Start it with:

```bash
cd ~/Code/moeLAR && nohup scripts/next_batch.sh > logs/next_batch.log 2>&1 &
tail -f logs/next_batch.log
```

Each step logs to `logs/<step>.log` (gitignored). After it finishes:

- `uv run python evals/compare.py evals/results/qwen3-4b-instruct-2507-4bit.json:4B evals/results/qwen3-4b-instruct-2507-4bit-head-v2.json:4B+head2 evals/results/qwen3-5-9b-mlx-4bit.json:9B evals/results/qwen3-5-9b-mlx-4bit-head.json:9B+head --metric acc` (and `ece`, `brier`, `ms`).
- Fill the `<!-- MOLAR_NUMBERS -->` placeholder in `examples/molar_triage/README.md`
  from `logs/molar-*.json`.
- Add a 9B section to `evals/RESULTS.md` with the ms/row and peak-memory cost next to
  the accuracy gain.

## Things to know

- `uv` lives at `~/Library/Python/3.9/bin/uv`; the project venv is Python 3.12.
- `origin` is SSH. A push from a fresh terminal will ask to accept github.com's host
  key once.
- Feature shards under `data/` and checkpoints are gitignored; regenerating the small
  shard takes eight minutes. The committed head is `evals/results/tier-b-qwen3-4b.head.npz`
  and its projection is `moelar.train.features.projection(2560, 512, seed=0)`.
- chaosnli borrows mnli's calibrator (no validation split). Any head that sharpens mnli
  hurts chaosnli's vote-distribution target; give it a flatter temperature or its own.
- Data policy is unchanged: no Jev-labeled data, no calls to the hosted API, Jev numbers
  quoted only from jev-bench's published run. See `DESIGN.md` section 8.

## Open ideas, not started

- Complement-consistency loss needs negated noul pairs generated in `moelar.train.build`.
- High-K configs (banking77, clinc150, ledgar) are the largest gaps after knowledge
  tasks; a shortlist-then-rerank pass or a description-aware prompt is the next lever
  after the 9B backbone.
- llama.cpp backend still needs a hardware pass.
