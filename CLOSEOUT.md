# Closeout to v0.1.0, planned 2026-09-25

Goal: a tagged GitHub release, `v0.1.0` on `moedex/moelars`, that serves the most accurate
configuration that answers the `load_cost.py` request in under 2 s warm on this machine
(128 GB M5 Max). The adapters go on the HF Hub under `moedex`. Check boxes as work lands; the
newest state wins over `HANDOFF.md`.

## Decisions (2026-09-25)

| question | decision |
|---|---|
| what "ship" means | tagged GitHub release, pinned default model + adapter(s) + calibrator, release notes |
| latency | accuracy first; the default must answer the `load_cost.py` request in under 2 s warm |
| two adapters (0.765) | confirm with a second seed; if the gain holds and fits the budget, it becomes the default |
| score-task regressions | document them as a known limitation; no fix in this release |
| where the weights go | HF Hub, `moedex` org, with model cards |
| training-data licenses | audit before publishing weights; this blocks the weights, not the code |
| code review | fix every open Medium (M3 to M10, M12, M13) before the tag |

Where things stand on 2026-09-25: a paired bootstrap over the committed test row dumps (10k
resamples, rows resampled within each config) puts averaging the two adapters at +1.3 over
attention-only alone, 95% CI [+0.7, +1.9], and attention-only at +1.9 over Jev's 0.733, CI
[+0.7, +3.1]. That covers test-row noise only, not seed variance, and treats Jev's number as exact.

## Order

```
Phase 0  license audit  ──┐        latency baseline (quiet machine, ~1 h)
                           ▼
Phase 1  GPU: seed-1 run (+ retrains if the audit drops a source)    ║  code: review fixes
                           ▼                                          ║
Phase 2  gate: does the two-adapter gain hold?  ──►  build two-adapter serving, or skip
                           ▼
Phase 3  as-served numbers, preset, weights on the Hub, docs, tag
```

The audit comes first because dropping a source means retraining, and the second-seed run
should be on the corpus we actually ship. The review fixes use no GPU and run alongside Phase 1.

---

## 1. Latency decides the architecture

- [x] Quiet machine (nothing else on the GPU). Run:
  ```
  M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
  uv run python scripts/load_cost.py $M30 $M30@checkpoints/lora-30b $M30@checkpoints/lora-30b-experts
  ```
  Record load time, peak memory, and warm ms per request in `evals/RESULTS.md`.
  **Done 2026-09-25:** 265 ms zero-shot, 299 ms attention, 384 ms experts; two adapters about 0.7 s.
- [ ] Two-adapter estimate for now: the sum of the two adapter rows. Measure it for real once §2
  builds it.
- [x] ~~Only if the 30B + LoRA takes over 2 s:~~ not needed (0.3 s). fuse the 4B here (`mlx_lm.fuse`, local snapshot
  path) and time `lora-4b-fused-bf16`, which brings 4B-to-30B routing (0.757 at 46%
  escalated) back into play.

**Rule:** the default is the most accurate configuration under 2 s. In order of preference:
the two 30B adapters averaged (if §2's gate passes), then the 30B + attention-only adapter,
then 4B routed to the 30B.

## 2. Two adapters on one 30B

### 2a. Confirm the gain is real (GPU, about 3 h)

- [ ] Train a second seed of the attention-only adapter on the corpus as it stands after the
  audit:
  `python -m moelars.train.lora --model $M30 --keys attn --seed 1 --records ... --out checkpoints/lora-30b-s1`
  (copy `scripts/queue_30b_lora.sh` into `scripts/queue_30b_lora_s1.sh`, with the
  `"improved": false` stop kept.)
- [ ] Suite with `--adapter checkpoints/lora-30b-s1 --tag lora-s1-rows --dump-rows`.
- [ ] Cascades with `evals/cascade.py`, plus the paired bootstrap from the closeout check
  (commit it as `evals/bootstrap.py`):
  - attn s1 alone, compared with attn s0 alone (the seed spread)
  - **attn s1 + experts** averaged (the replication)
  - **attn s0 + attn s1** averaged (the control: is the gain from mixing attention-only with
    experts, or just from ensembling any two runs?)

**Gate:** attn s1 + experts beats attn s1 alone with a bootstrap 95% CI above zero, and the
choice of averaging over switching is made on validation. If the control does as well as the
mixed pair, ship whichever pair is cheaper to serve (two attention-only adapters are smaller).
If the gate fails, the default is one adapter (s0 or s1, whichever has the better validation
macro) and averaging goes in the release notes as future work.

### 2b. Serving, only if the gate passes

- [ ] `MLXBackend` takes several adapters: load the base once, build LoRA layers for the
  union of the adapters' keys (experts ⊇ attention), keep each adapter's weights in memory,
  and swap them with `model.load_weights(..., strict=False)` between passes. Expert keys stay
  zero while the attention-only adapter is loaded.
- [ ] `Engine` averages the adapters' calibrated probabilities per question (the blend in
  `evals/cascade.py`), then applies constraints once on the average.
- [ ] CLI: `--adapter` can be repeated. Tests: two identity adapters must match one; the
  average must equal the per-pass mean.
- [ ] Re-run §1's `load_cost.py` with both adapters and check it is under 2 s.
- [ ] Suite with both adapters live (not simulated) must reproduce the cascade number within
  ±0.5.

## 3. Score-task regressions: document them

- [ ] A table in `evals/RESULTS.md` and the release notes with every config that the shipped
  default scores below the zero-shot 30B. Currently that includes stsb 0.425 → 0.350 and mmlu
  0.780 → 0.750, and the experts adapter drops stsb further to 0.275.
- [ ] State the headline honestly: civil_comments is worth about 1.8 macro points; without it
  the lead over Jev is about 1 point, inside the noise. Our numbers use 200 test rows per
  config; Jev's use full splits.
- [ ] Ordinal loss and more score sources go on the post-release list (below).

## 4. Code-review Mediums (no GPU; alongside Phase 1)

Each fix gets a test and a status line in `CODEBASE-REVIEW.md`, same format as M1.

- [x] **M3** Reject a request `model` that isn't the loaded model's ID or its documented
  alias (`moelars` or the preset name); 400 `invalid_request`.
- [x] **M4** Split the prompt at the offset where the sentinel was inserted, not at the first
  match; test with the sentinel inside state.
- [x] **M5** Apply constraints until nothing changes (with an iteration cap) and check the
  invariants; round only at the end; test the exclusive+complement case from the review.
- [x] **M6** Keep units after 24 in every ablated state (or decline evidence past 24 units
  with a warning); test with a 30-unit state.
- [x] **M7** Pass option offsets from `render` to the pointer head instead of re-parsing
  `A) `. (No escaping: prompts stay byte-identical, so no suite re-run is needed for M4 or M7.)
- [x] **M8** Reject `multi` examples in eval/calibrate with a clear error (per-option labels
  are post-release).
- [x] **M9** Use the normalized soft target for nouls in metrics and Platt fitting.
- [x] **M10** `coverage_at_error` evaluates groups of equal confidence together; test
  `[0.9, 0.9]` / `[1, 0]`.
- [x] **M12** Extraction deletes stale `heldout.npz` / `test.npz` and writes a manifest
  fingerprint; consumers check it.
- [x] **M13** Fingerprint suite inputs per config (rows, backend, adapter, head, template, data
  hash); a mismatch or a missing row dump means recomputing.
- [ ] After the M4, M5 and M7 fixes, re-run the suite for the shipped default and confirm the
  headline hasn't moved by more than noise (render changes affect every prompt).

## 5. Data license audit (Phase 0, blocks the weights)

- [ ] A per-source table in `DESIGN.md` section 8: upstream dataset, license, whether it
  allows derivative weights, and whether commercial use is OK. Known items to check:
  - `tasksource-jev/anli/*` (3,300 of 4,000 tasksource train rows): ANLI is, as I recall,
    CC BY-NC 4.0.
  - `jev-bench/yelp5`: Yelp Dataset terms.
  - `jev-bench/ledgar`, `jev-bench/fever_evidence`: CC BY-SA (share-alike).
  - the rest of the 21 jev-bench configs, `babi_nli`, `glue/mnli`, and Open-Jev (CC0) to
    confirm.
- [ ] **Decide per source:** keep it (and note it in the model card), mark the weights as
  non-commercial, or drop it and retrain. Dropping any source means retraining both adapters
  (135 + 169 min plus suites) before §2a, and redoing §3's numbers.
- [ ] Qwen3-30B-A3B base is Apache 2.0; the model card credits it.

## 6. Release (Phase 3)

- [ ] **Calibrator as served.** Per-config calibrators can't be used for arbitrary requests.
  Fit one pooled calibrator over all configs' validation rows for the shipped default and add
  `--calibration FILE` to `evals/run_suite.py`. Report the pooled number as the headline and
  per-config as secondary. (Raw and per-config calibrated accuracy differ only on
  civil_comments, so expect about 0.75.)
- [ ] `moelars serve --preset 30b` (also `30b-duo` if §2 passes) resolves the model, the
  adapter(s) from the Hub (`huggingface_hub.snapshot_download`) and the pooled calibrator.
  `--backend mock` stays the default with no preset. Add a preset smoke test that uses the
  mock backend in CI.
- [ ] HF Hub: `moedex/moelars-qwen3-30b-a3b-lora-attn` (+ `-experts` or `-attn-s1`), each with
  `adapter_config.json`, safetensors, the calibrator, and a model card: base model, recipe and
  seed, training sources and licenses from §5, the jev-bench table, known limitations, and the
  data policy (no Jev-labeled data).
- [ ] Docs: README quickstart using the preset, the headline table, the civil_comments caveat,
  and "MLX only; llama.cpp untested on hardware". Update `HANDOFF.md` and `evals/RESULTS.md`.
- [ ] Bump `version` to 0.1.0 in `pyproject.toml` and `src/moelars/__init__.py`, and add a
  `CHANGELOG.md` entry.
- [ ] Final checks on a clean checkout: `uv sync`, `ruff`, `pytest` (MLX tests here), CI green
  on `main`, and a quiet `load_cost.py` run with the preset under 2 s.
- [ ] Tag `v0.1.0`, GitHub release with notes (headline, latency, known limitations, links to
  the Hub).

## Not in this release

- Ordinal-aware loss and more score sources (the stsb and mmlu regressions).
- Molar Triage numbers (`examples/molar_triage/README.md` placeholder).
- Per-option labels for `multi` in eval/calibrate (M8's full fix).
- Gating the pointer head on K, shortlist-then-rerank for high K, complement-consistency loss.
- A hardware pass for the llama.cpp backend.
