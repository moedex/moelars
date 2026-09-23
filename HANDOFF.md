# Handoff, 2026-09-22 evening

Everything below is committed. Nothing of ours is running.
`origin` is `git@github.com:moedex/moelars.git`.

## State in one paragraph

moe-LARS serves the System One wire shape from a local model with no text generation. On
jev-bench (22 configs, 200 test rows each, per-config calibration on every side),
Qwen3-4B-Instruct with a LoRA adapter trained through the label readout reaches **0.731**
macro accuracy against Jev's published 0.733, with better Brier (0.317 against 0.349)
and ECE (0.077 against 0.113). The gain is on question forms the training corpus covers;
held-out forms are flat except stsb, which the adapter breaks. Qwen3-30B-A3B zero-shot is
0.680 and strongest exactly where the adapted 4B is weakest. Full tables and readings:
`evals/RESULTS.md`, newest section last.

## What happened in this batch

- `scripts/queue_lora.sh` ran end to end (14:50 to 20:20): 30B smoke passes under the new
  check; LoRA on the full corpus (113 minutes, about 27 GB peak); suite with the adapter
  (0.731); features and head on the adapted model (the head adds nothing, 0.729); 30B-A3B
  Tier A suite (0.680).
- `moelars.train.residual` now keeps the untrained head (the backbone exactly) as the
  checkpoint to beat. Before, it always saved some epoch, even one worse than no head.
- `evals/run_suite.py --dump-rows` writes calibrated per-row probabilities for test and
  validation; `evals/cascade.py` simulates an `escalate_to` cascade from two such dumps,
  with one confidence floor chosen on validation and applied to test. **Not run on real
  models yet**: it needs both suites rerun with `--dump-rows`.
- The adapter was fused with `mlx_lm.fuse` two ways, 4-bit re-quantized
  (`checkpoints/lora-4b-fused-q4`, 2.1 GB) and bf16 (`checkpoints/lora-4b-fused-bf16`,
  7.5 GB), and checked against the unfused adapter on 7 configs. bf16 matches the
  adapter and runs 25 to 40 percent faster; 4-bit loses about half of the adapter's gain
  and should not be served. Neither fused model is committed; both regenerate in a minute.
  `mlx_lm.fuse` needs the local snapshot path as `--model` (the hub cache is incomplete
  and the environment is offline).

## Next

1. **Cascade numbers.** Rerun the adapted 4B (`checkpoints/lora-4b-fused-bf16`) and the
   30B-A3B suites with `--dump-rows --tag rows` (about an hour and 1.5 hours with the
   extra validation pass), then `evals/cascade.py <4B rows dir> <30B rows dir>`.
2. **stsb under LoRA** (0.395 to 0.235): ordinal-aware loss for score tasks, more score
   sources in `moelars.train.build`, or a gentler adapter. Must be understood before LoRA
   is the default.
3. Molar Triage numbers (`scripts/queue_30b.sh` has the loop; the `<!-- MOLAR_NUMBERS -->`
   placeholder in `examples/molar_triage/README.md`) and `scripts/load_cost.py` for 4B,
   4B fused bf16, and 30B on a quiet machine.

## Things to know

- `uv` lives at `~/Library/Python/3.9/bin/uv`; the project venv is Python 3.12.
- The user often shares the dev box's GPU with other work. Treat in-queue ms/row as rough;
  quote latency only from a quiet-machine `load_cost.py` run. Accuracy comes first.
- HF cache weights sit behind symlinks into xet blob folders; `du` on a model directory
  shows megabytes. Use `ls -laL snapshots/*/`.
- LoRA's `scale` multiplies every update; on a toy model lr 1e-2 oscillated and 3e-3
  converged. The default is 2e-5 at scale 20. The adapter (`checkpoints/lora-4b`, 66 MB)
  is not committed; its history and config are in `evals/results/lora-4b.*`.
- chaosnli borrows mnli's calibrator (no validation split); under LoRA it fell 4.5 points,
  possibly from the borrowed temperature.
- civil_comments' calibrated 0.930 is the majority baseline and is worth about 1.8 macro
  points wherever moe-LARS leads; the headline flatters us by about that much.
- The 30B scores 0.110 on helpsteer2_verbosity (all mass on one end of the scale);
  calibration cannot fix an argmax. Not investigated.
- Data policy is unchanged: no Jev-labeled data, no calls to the hosted API, Jev numbers
  quoted only from jev-bench's published run. See `DESIGN.md` section 8.

## Open ideas, not started

- Gate the pointer head on K: on LoRA it helps high-K routing and hurts elsewhere. Read
  off test, so check on held-out data first.
- Shortlist-then-rerank for high-K configs (banking77, clinc150, massive, ledgar).
- Complement-consistency loss needs negated noul pairs generated in `moelars.train.build`.
- llama.cpp backend still needs a hardware pass.
