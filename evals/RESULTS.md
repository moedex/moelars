# Results

All MoeLAR numbers are measured in this repository. Jev numbers are quoted from the
jev-bench maintainers' published run of `jev-1.13.0` on the full test splits, recorded
in `evals/jev_published.json`. See `DESIGN.md` section 8 for why we never call that API.

## 2026-09-22: Tier A, all 22 jev-bench configs, Qwen3-4B-Instruct-2507 4-bit on MLX

Hardware: Apple M5 Max, 64 GB. Backend `mlx`, mlx-lm 0.31.3. 200 test rows per config,
one question per request. Calibration fitted on 200 validation rows of the same config:
a temperature for choice and score, a Platt pair on the raw yes-minus-no logit for
noul. chaosnli has no validation split and borrows mnli's calibrator. Total wall time
for the suite was 30 minutes. Full table: `evals/results/qwen3-4b-instruct-2507-4bit.md`.

### Macro averages, calibrated

| scope | n | MoeLAR acc | Jev acc | MoeLAR ECE | Jev ECE | MoeLAR Brier | Jev Brier |
|---|---|---|---|---|---|---|---|
| all | 22 | 0.662 | 0.733 | **0.088** | 0.113 | 0.404 | 0.349 |
| choice | 9 | 0.657 | 0.770 | **0.084** | 0.112 | 0.386 | 0.345 |
| score | 6 | 0.466 | 0.503 | **0.116** | 0.197 | **0.645** | 0.662 |
| noul | 7 | 0.837 | 0.881 | 0.069 | 0.043 | 0.222 | 0.085 |

### Where MoeLAR is ahead

| config | MoeLAR acc | Jev acc | note |
|---|---|---|---|
| chaosnli | 0.685 | 0.615 | scored against 100-annotator vote shares; Brier 0.165 vs 0.583 |
| helpsteer2_verbosity | 0.585 | 0.341 | Brier 0.619 vs 0.793 |
| civil_comments | 0.930 | 0.729 | see caveat below |

### Where the gap is largest

| config | MoeLAR acc | Jev acc | why |
|---|---|---|---|
| mmlu | 0.670 | 0.923 | world knowledge, set by the 4B base model |
| strategyqa_closed | 0.570 | 0.785 | closed-book multi-hop, same cause |
| clinc150 (K=151) | 0.725 | 0.893 | high-cardinality routing |
| mnli | 0.725 | 0.883 | |
| banking77 (K=77) | 0.665 | 0.796 | high-cardinality routing |

### Reading

- **Calibration is largely solved by fitting.** Calibrated ECE beats Jev's macro on every
  primitive except noul, and on the score primitive the calibrated Brier is also ahead.
  Fitted temperatures ran from 4 to 8, which is how overconfident raw instruct logits are.
  Temperature never changes an answer.
- **Platt on nouls changes answers, and it matters.** fever_evidence +1.0, sms_spam
  +4.0, strategyqa_closed +2.0, strategyqa_grounded +2.5 points, and civil_comments
  from 0.510 to 0.930. Temperature alone cannot move a decision across 0.5. This is the
  case the Laya benchmark documented and the reason the Platt path exists.
- **Accuracy is the whole problem, and it is the base model.** The macro accuracy of
  0.662 happens to equal the jev-bench maintainers' own zero-shot Qwen3.5-4B figure.
  Their LoRA plus residual head on the same 4B reaches 0.747, above Jev. Their
  zero-shot Qwen3.5-9B alone reaches 0.689. The gap scales with option count and with
  how much world knowledge the question needs.
- **Coverage at a 5% error budget** is the operator number. It ranges from 0.96 on
  civil_comments and 0.94 on fever_evidence down to zero on stsb. Do not put a 4B
  zero-shot model behind an auto-approve gate on high-K routing yet.

### Caveats

- 200 rows per split against Jev's 1000. Expect a few points of noise per config; the
  macro over 22 configs is more stable.
- **civil_comments**: 93% of the test rows are "not toxic". The calibrated 0.930 equals
  the majority baseline. The fitted Platt bias is -3.75, which says the raw model called
  nearly everything toxic and calibration pushed it back to the base rate. Jev's 0.729
  is below the majority baseline, so neither number says much about toxicity detection.
  Read this config through its Brier and coverage columns instead.
- **Brier definitions may differ for noul.** MoeLAR sums squared error over both
  outcomes, which is twice the single-probability Brier. If jev-bench reports the
  single-probability form for noul, halve the MoeLAR noul Brier for comparison
  (0.222 becomes 0.111 against Jev's 0.085). Choice and score sums are standard.
- Older Qwen3 (2507) rather than Qwen3.5, because that is what mlx-community had
  quantized on the day.

## 2026-09-22 (earlier): first three configs

Superseded by the full table above; kept in git history.
