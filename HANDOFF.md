# Handoff, 2026-09-22 afternoon

Everything below is committed. Nothing of ours is running; the GPU was lent to another job
at 13:25. `origin` is `git@github.com:moedex/moelar.git`; this batch is not pushed yet.

## State in one paragraph

MoeLAR serves the System One wire shape from a local model with no text generation. On
jev-bench (22 configs, 200 test rows each, per-config calibration on every side), Qwen3-4B
Tier A is 0.662 macro accuracy, 4B plus pointer head v2 is **0.689**, Jev's published run is
0.733. Our ECE is 0.073 against Jev's 0.113. The head's gain is all on question forms its
training corpus covers; held-out sources are flat. A Qwen3.5-9B Tier A run, stopped after
11 configs, added about 2 points over the 4B at two to three times the cost per row, so
dense scale is not the default path. Full tables and readings: `evals/RESULTS.md`.

## What happened in this batch

- Head v2 on the full clean corpus (14,285 records): +2.7 macro over Tier A (v1: +1.8),
  mostly high-K routing. Artifacts `evals/results/tier-b-qwen3-4b-v2.*`.
- Qwen3.5 backend bug fixed: `ArraysCache` snapshots were aliased, so each restored row
  mutated the prefix snapshot for the next. Tested with the real mlx-lm cache classes.
- 9B suite stopped by hand at 11 of 22 configs (`evals/results/qwen3-5-9b-mlx-4bit.*`).
  `run_suite.py` skips configs already in its result file, so rerunning the same command
  resumes at config 12 if the full 9B number is ever wanted.
- 9B feature extraction and head were cut. `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`
  (MoE, about 3B active, 17 GB) is downloaded as the cascade target or larger backbone.
- Smoke check restructured: a strict check that restoring the snapshot equals prefill then
  suffix on one cache (isolates cache bugs), plus a split-versus-full check whose bound is
  two bf16 steps at the logits' magnitude. The 30B failed the old fixed 0.15 bound at
  0.375 on logits near 35 (bf16 step 0.25); it has not been rerun under the new check.
- **LoRA through the label readout** (`moelar.train.lora`), the half of Tier B that
  `DESIGN.md` specifies and we had not built. Cross-entropy plus Brier on the K label
  logits at the answer position, options reshuffled per epoch, sources held out with the
  same sampling as `extract`, best held-out Brier checkpoint saved in mlx-lm's adapter
  format. `--adapter` works on serve, eval, calibrate, the suite, and extraction. Unit
  tested on a tiny random Qwen3; **never run on a real model yet**.

## Next: run `scripts/queue_lora.sh` when the GPU is free

```bash
cd ~/Code/moeLAR && nohup scripts/queue_lora.sh > logs/queue_lora.log 2>&1 &
```

Order: 30B smoke under the new check; a 400-record LoRA probe (read its log for steps per
minute and whether memory is sane before trusting the full run's duration); LoRA on the
full corpus; suite with the adapter alone; features, head, and suite for adapter plus head;
then the 30B-A3B Tier A suite. The jev-bench maintainers' LoRA plus residual head on the
same 4B reaches 0.747, above Jev; that is the number to compare against.

Still to do after that: Molar Triage numbers (`scripts/queue_30b.sh` has the loop; the
`<!-- MOLAR_NUMBERS -->` placeholder in `examples/molar_triage/README.md`), and
`scripts/load_cost.py` for 4B, 9B, and 30B on a quiet machine.

## Things to know

- `uv` lives at `~/Library/Python/3.9/bin/uv`; the project venv is Python 3.12.
- The user often shares the dev box's GPU with other work. Treat in-queue ms/row as rough;
  quote latency only from a quiet-machine `load_cost.py` run. Accuracy comes first.
- HF cache weights sit behind symlinks into xet blob folders; `du` on a model directory
  shows megabytes. Use `ls -laL snapshots/*/`.
- LoRA's `scale` multiplies every update; on a toy model lr 1e-2 oscillated and 3e-3
  converged. The default is 2e-5 at scale 20.
- chaosnli borrows mnli's calibrator (no validation split).
- civil_comments' calibrated 0.930 is the majority baseline and is 1.8 of the 3.2 macro
  points where MoeLAR beats Jev; the headline flatters us by about that much.
- Data policy is unchanged: no Jev-labeled data, no calls to the hosted API, Jev numbers
  quoted only from jev-bench's published run. See `DESIGN.md` section 8.

## Open ideas, not started

- `escalate_to` cascade: answer on the 4B, send low-confidence rows to the 30B-A3B. Our
  calibration is what makes the confidence floor meaningful.
- Shortlist-then-rerank for high-K configs (banking77, clinc150, massive, ledgar).
- Complement-consistency loss needs negated noul pairs generated in `moelar.train.build`.
- Ordinal-aware loss for score tasks, which neither head moved.
- llama.cpp backend still needs a hardware pass.
- Rename to moe-LARS (`moelars` in code), agreed with the user but deferred until they
  say go; the spec is in the assistant's project memory.
