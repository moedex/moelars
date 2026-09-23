# Handoff, 2026-09-23 morning

Everything below is committed and pushed to `main` on `git@github.com:moedex/moelars.git`
(renamed from moedex/moelar; GitHub redirects the old URL). Nothing of ours is running.
The next job, a full LoRA on Qwen3-30B-A3B, is to be run **on another machine**; the steps
below assume a fresh clone there.

## State in one paragraph

moe-LARS (Moe Limited but Accurate Response System; `moelars` in code) serves the System
One wire shape from a local model with no text generation. On jev-bench (22 configs, 200
test rows each, per-config calibration on every side), Qwen3-4B-Instruct with a LoRA
adapter fused in bf16 is **0.731** macro accuracy against Jev's published 0.733, with better
Brier (0.317 against 0.349). Routing the 4B's low-confidence rows to a zero-shot
Qwen3-30B-A3B and averaging the two models' probabilities there reaches **0.746** (Brier
0.308) with a quarter of rows escalated, every floor chosen on validation only. Full
tables: `evals/RESULTS.md`, newest section last.

## What happened since the last handoff

- Package, CLI, wire strings, and repo renamed to moe-LARS / `moelars`, clean break, no
  aliases. Committed result and calibration files keep the old name they ran under.
- Merged the user's `fix/mlx-cache-limit` (MLX buffer cache capped at 4 GB,
  `MOELARS_MLX_CACHE_GB`) and `feat/evidence-features` (numeric evidence fused with noul
  answers via `moelars.features`), plus `CODEBASE-REVIEW.md` (14 medium findings, one
  low; its paths predate the rename, `src/moelar/` is now `src/moelars/`).
- The bf16-fused 4B adapter matches the unfused one on all 22 configs; 4-bit fusion loses
  half the gain. `checkpoints/lora-4b-fused-bf16` (7.5 GB) is on the old machine only.
- Both models' suites rerun with `--dump-rows`; `evals/cascade.py` gained `blend` and
  per-kind floors. The row dumps are committed (`evals/results/rows/`, 16 MB), so routing
  can be recomputed anywhere without a GPU.
- `moelars.train.lora --keys attn|attn+experts|all`, and a patch that makes mlx-lm's
  `qwen3_moe` trainable (its router indices lacked `stop_gradient`). 30B probes: attention
  only is practical (6.7M params, 5.5 steps/min, 32 GB peak, held-out improving);
  attention plus all experts (422M params) swapped and was stopped on a 64 GB machine.

## Next: full attention-only LoRA on the 30B-A3B, on the other machine

### 1. Set up

```bash
git clone git@github.com:moedex/moelars.git && cd moelars
uv sync --extra dev --extra mlx --extra evals
uv run pytest -q                                   # 67 tests, a few seconds
uv run hf download mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit   # about 17 GB
```

### 2. Copy the data from the old machine, do not rebuild it

`data/` and `evals/data/` are gitignored. Rebuilding with `moelars.train.build` and
`evals/fetch_jevbench.py` can pick up newer dataset revisions and change the training
corpus or the held-out split, which would make the 30B adapter incomparable with the 4B.
Copy them instead (about 62 MB):

```bash
rsync -a <old-machine>:Code/moeLAR/data/ data/
rsync -a <old-machine>:Code/moeLAR/evals/data/ evals/data/
```

Check: `data/train/manifest.json` exists, `ls evals/data/*.test.jsonl | wc -l` is 22,
and the LoRA log's first line reads `records: 14285 -> train 12685 / heldout 1600`, the
same split as the 4B adapter.

### 3. Recommended before the run: codebase review M11

If no held-out evaluation beats the untrained model, `moelars.train.lora` still saves
the last trained adapter as if it were selected. `moelars.train.residual` already starts
selection from the untrained model; `lora.py` should do the same (record the step-0
held-out Brier as the score to beat, and save an adapter only when a checkpoint beats
it, or mark the run as not improved). Not fixed yet; about ten minutes plus a test.

### 4. Run

```bash
nohup scripts/queue_30b_lora.sh > logs/queue_30b_lora.log 2>&1 &
```

Steps, each logging to `logs/<step>.log`:

1. `lora-30b`: attention-only LoRA, full corpus, one epoch, about 1,100 steps. On the
   old 64 GB machine that is 3.5 to 4 hours at 32 GB peak; held-out metrics print every
   500 steps. The script sets `MOELARS_MLX_CACHE_GB=16` for training.
2. `suite-30b-lora`: all 22 configs with the adapter unfused (a bf16 fuse of the 30B
   would be about 60 GB), per-config calibration, and per-row dumps. About 1.5 hours.
3. `cascade-30b-lora`: routing with the committed 4B dumps as primary and the new
   30B-plus-LoRA dumps as fallback. Seconds.

### 5. Read the results

- `evals/results/qwen3-30b-a3b-instruct-2507-4bit-lora-rows.md`: the adapted 30B alone.
  Compare with 0.731 (4B + LoRA), 0.680 (30B zero-shot), and 0.733 (Jev). Watch stsb
  (0.235 under the 4B's LoRA, 0.425 for the zero-shot 30B) and mmlu (0.780 zero-shot).
- `logs/cascade-30b-lora.log`: the routing table. Compare with 0.746 for the zero-shot
  30B as fallback. If the adapted 30B alone beats the routed pair, the next question is
  latency (`scripts/load_cost.py` on a quiet machine), not accuracy.
- Commit the result JSON/MD, `calibration/*lora-rows*`, the row dumps, and
  `evals/results/lora-30b.{history,adapter_config}.json` (copy them out of
  `checkpoints/`, which is ignored), then write the section in `evals/RESULTS.md`.

### If the other machine has more memory

The 30B probe of attention plus all experts (`--keys attn+experts`) was stopped only
because a 64 GB machine swapped. With 128 GB it may run; probe it first with
`--limit 400 --eval-every 50` and compare steps per minute before committing hours.

## After that

1. **stsb under LoRA** (0.395 zero-shot to 0.235 on the 4B): ordinal-aware loss for score
   tasks, more score sources in `moelars.train.build`, or a gentler adapter.
2. Codebase review findings M1 to M14 and L1: request budgets (M1) and the blocking
   event loop (M2) matter before anyone else calls the server.
3. Store an example ID in row dumps so `evals/cascade.py` checks alignment by ID (M14).
4. Molar Triage numbers (`scripts/queue_30b.sh` has the loop; the `<!-- MOLAR_NUMBERS -->`
   placeholder in `examples/molar_triage/README.md`) and `scripts/load_cost.py` for 4B,
   4B fused bf16, and 30B on a quiet machine.

## Things to know

- The old machine: `uv` at `~/Library/Python/3.9/bin/uv`, `gh` at `/opt/homebrew/bin/gh`,
  checkout at `~/Code/moeLAR` (folder name unchanged by the rename). Queue scripts there
  default `UV` to that path; `scripts/queue_30b_lora.sh` defaults to `uv` on PATH.
- The user shares GPUs with other work. Treat in-queue ms/row as rough; quote latency
  only from a quiet-machine `load_cost.py` run. Accuracy comes first.
- HF cache weights sit behind symlinks into xet blob folders; `du` on a model directory
  shows megabytes. Use `ls -laL snapshots/*/`. If the environment is offline,
  `mlx_lm.fuse` needs the local snapshot path as `--model`.
- LoRA's `scale` multiplies every update. Defaults are lr 2e-5, rank 8, scale 20.
- chaosnli borrows mnli's calibrator (no validation split).
- civil_comments' calibrated 0.930 is the majority baseline and is worth about 1.8 macro
  points wherever moe-LARS leads; the headline flatters us by about that much.
- The zero-shot 30B scores 0.110 on helpsteer2_verbosity (all mass on one end of the
  scale); calibration cannot fix an argmax.
- Data policy is unchanged: no Jev-labeled data, no calls to the hosted API, Jev numbers
  quoted only from jev-bench's published run. See `DESIGN.md` section 8.

## Open ideas, not started

- Gate the pointer head on K: on LoRA it helps high-K routing and hurts elsewhere. Read
  off test, so check on held-out data first.
- Shortlist-then-rerank for high-K configs (banking77, clinc150, massive, ledgar).
- Complement-consistency loss needs negated noul pairs generated in `moelars.train.build`.
- llama.cpp backend still needs a hardware pass.
