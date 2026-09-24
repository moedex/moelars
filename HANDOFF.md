# Handoff, 2026-09-23 evening

Everything below is committed and pushed to `main` on `git@github.com:moedex/moelars.git`.
Nothing of ours is running.

## State in one paragraph

moe-LARS (Moe Limited but Accurate Response System; `moelars` in code) serves the System
One wire shape from a local model with no text generation. On jev-bench (22 configs, 200
test rows each, per-config calibration on every side), **Qwen3-30B-A3B with an
attention-only LoRA is 0.752 macro accuracy against Jev's published 0.733**, Brier 0.295
against 0.349, and 0.743 against 0.733 without civil_comments. Qwen3-4B with its LoRA is
0.731. Routing the 4B to the adapted 30B reaches 0.757 at 46% escalated, inside the noise
of the 30B alone. Full tables: `evals/RESULTS.md`, newest section last.

## What happened today (afternoon)

- Codebase review M11 fixed: if no checkpoint beats the untrained model, `moelars.train.lora`
  saves the untrained adapter (identity) with `"improved": false`, keeps the last state in
  `<out>.last/`, and clears a stale adapter in `--out` first. `scripts/queue_30b_lora.sh`
  stops before the suite in that case. One new test (68 total).
- Full attention-only LoRA on the 30B-A3B: 1095 steps, 135 minutes on this machine, 33 GB
  peak, held-out Brier 0.742 to 0.454 (step 1000 selected, `"improved": true`).
- Suite with the adapter (unfused) and row dumps, and routing against the 4B dumps:
  committed under `evals/results/` and `calibration/`.
- Probes here: attention plus every expert (422M parameters) now fits, 4.0 steps/min at
  93 GB peak, and its 31-step held-out Brier (0.396) beats attention only (0.437) on 55
  rows.

## Next

1. **Latency decides the architecture.** Accuracy no longer separates the 30B + LoRA alone
   from the routed pair. Run `scripts/load_cost.py` on this machine (quiet) for the 4B
   fused bf16 (copy it from the old machine, or fuse it again here), the 30B,
   and the 30B with `--adapter checkpoints/lora-30b`, then decide what `moelars serve`
   defaults to.
2. **Full attention-plus-experts LoRA on the 30B** (`--keys attn+experts`, same corpus,
   about 4.6 hours at 93 GB peak, `MOELARS_MLX_CACHE_GB=16`), then the suite and
   `evals/cascade.py` as in `scripts/queue_30b_lora.sh` with a new `--out` and `--tag`.
   It is only worth keeping if it beats 0.752 by more than the ~0.7-point noise.
3. **stsb and mmlu under LoRA** regress on both models (30B: stsb 0.425 to 0.350, mmlu
   0.780 to 0.750): ordinal-aware loss for score tasks, more score sources in
   `moelars.train.build`, or a gentler adapter.
4. Codebase review findings M3 to M10, M12 and M13 (M1, M2, M11, M14 and L1 are fixed;
   see the status lines in `CODEBASE-REVIEW.md`).
5. Molar Triage numbers (`scripts/queue_30b.sh` has the loop; the `<!-- MOLAR_NUMBERS -->`
   placeholder in `examples/molar_triage/README.md`).

## Things to know

- Two machines. This one (128 GB M5 Max): checkout at `~/Code/personal/moelars`, `uv` on
  PATH, 30B weights in the HF cache, `checkpoints/lora-30b` (the selected adapter) and the
  two probe adapters. The old 64 GB one: `uv` at `~/Library/Python/3.9/bin/uv`, checkout at
  `~/Code/moeLAR`, and `checkpoints/lora-4b-fused-bf16`. Older queue scripts default `UV`
  to the old path; run them here with `UV=uv`.
- `data/` and `evals/data/` came from the old machine as a zip (same 14285-record corpus
  and split); do not rebuild them.
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
