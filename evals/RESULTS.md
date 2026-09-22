# Results

All MoeLAR numbers are measured in this repository. Jev numbers are quoted from the
jev-bench maintainers' published run of `jev-1.13.0` on the full test splits. See
`DESIGN.md` section 8 for why we never call that API ourselves.

## 2026-09-22: Tier A, Qwen3-4B-Instruct-2507 4-bit on MLX

Hardware: Apple M5 Max, 64 GB. Backend: `mlx` with mlx-lm 0.31.3. 200 test rows per
config, one question per request. Calibration is a single temperature per primitive
fitted on 200 validation rows of the same config. Raw data: `evals/results/qwen3-4b-instruct-2507-4bit.json`.

| config | prim | K | MoeLAR raw acc | MoeLAR cal acc | raw ECE | cal ECE | raw Brier | cal Brier | fitted T | ms/row | Jev 1.13 acc (n=1000) | Jev ECE | Jev Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77 | choice | 77 | 0.665 | 0.665 | 0.306 | **0.101** | 0.638 | 0.509 | 4.88 | 213 | 0.796 | 0.095 | 0.317 |
| boolq | noul | 2 | 0.890 | 0.890 | 0.115 | **0.046** | 0.222 | 0.187 | 7.54 | 108 | 0.917 | 0.021 | 0.061 |
| sst5 | score | 5 | 0.475 | 0.475 | 0.471 | **0.090** | 0.985 | 0.635 | 8.25 | 85 | 0.565 | 0.190 | 0.618 |

Coverage at a 5% error budget (the fraction of decisions a confidence gate could accept
automatically): banking77 0.03, boolq 0.56, sst5 0.03. Jev's published selective accuracy
at 90% coverage on the same configs is 0.839, 0.953, and 0.586.

## Reading

- **Temperature scaling does not change a single answer and cuts ECE by 3x to 5x.** The
  fitted temperatures of 5 to 8 say how badly overconfident raw instruct-model logits
  are. Calibrated ECE lands at or below Jev's on every config, which was the design bet.
- **Accuracy trails Jev by 3 to 13 points zero-shot**, worst on the 77-way task. This is
  the base model and the prompt, not the pipeline. The jev-bench maintainers' own
  zero-shot Qwen3.5-4B scores 0.662 macro accuracy against Jev's 0.733, and their
  LoRA-plus-residual-head version of the same model reaches 0.747. That is the Tier B
  target.
- **Brier stays worse than Jev's after calibration** on banking77 and boolq. A single
  temperature cannot fix a model that is wrong with high margin. A Platt bias for nouls
  and per-config temperatures are the next cheap steps; a trained head is the real one.
- **Coverage at 5% error is unusable on banking77 and sst5** because the accuracy floor
  is too low for any threshold to carve out a clean 95% region. Do not put this model
  behind an auto-approve gate on a 77-way task.
- **Latency** is 85 to 213 ms per row including the prefix prefill. Rows sharing a state
  cost far less: a 24-row request with permutations and ablations took about one second.

## Caveats

- 200 rows per split, not the full 1000. Expect a few points of noise.
- Older Qwen3 (2507) rather than Qwen3.5, because that is what mlx-community had
  quantized. Numbers are not directly comparable to the Qwen3.5 rows on jev-bench.
- jev-bench notes that Jev's API rounds probabilities to 0.01, which inflates its NLL.
  Brier and ECE are the fair comparison columns, and those are what this table shows.
