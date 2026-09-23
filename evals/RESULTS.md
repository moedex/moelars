# Results

All moe-LARS numbers are measured in this repository. Jev numbers are quoted from the
jev-bench maintainers' published run of `jev-1.13.0` on the full test splits, recorded
in `evals/jev_published.json`. See `DESIGN.md` section 8 for why we never call that API.

## 2026-09-22: Tier A, all 22 jev-bench configs, Qwen3-4B-Instruct-2507 4-bit on MLX

Hardware: Apple M5 Max, 64 GB. Backend `mlx`, mlx-lm 0.31.3. 200 test rows per config,
one question per request. Calibration fitted on 200 validation rows of the same config:
a temperature for choice and score, a Platt pair on the raw yes-minus-no logit for
noul. chaosnli has no validation split and borrows mnli's calibrator. Total wall time
for the suite was 30 minutes. Full table: `evals/results/qwen3-4b-instruct-2507-4bit.md`.

### Macro averages, calibrated

| scope | n | moe-LARS acc | Jev acc | moe-LARS ECE | Jev ECE | moe-LARS Brier | Jev Brier |
|---|---|---|---|---|---|---|---|
| all | 22 | 0.662 | 0.733 | **0.088** | 0.113 | 0.404 | 0.349 |
| choice | 9 | 0.657 | 0.770 | **0.084** | 0.112 | 0.386 | 0.345 |
| score | 6 | 0.466 | 0.503 | **0.116** | 0.197 | **0.645** | 0.662 |
| noul | 7 | 0.837 | 0.881 | 0.069 | 0.043 | 0.222 | 0.085 |

### Where moe-LARS is ahead

| config | moe-LARS acc | Jev acc | note |
|---|---|---|---|
| chaosnli | 0.685 | 0.615 | scored against 100-annotator vote shares; Brier 0.165 vs 0.583 |
| helpsteer2_verbosity | 0.585 | 0.341 | Brier 0.619 vs 0.793 |
| civil_comments | 0.930 | 0.729 | see caveat below |

### Where the gap is largest

| config | moe-LARS acc | Jev acc | why |
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
- **Brier definitions may differ for noul.** moe-LARS sums squared error over both
  outcomes, which is twice the single-probability Brier. If jev-bench reports the
  single-probability form for noul, halve the moe-LARS noul Brier for comparison
  (0.222 becomes 0.111 against Jev's 0.085). Choice and score sums are standard.
- Older Qwen3 (2507) rather than Qwen3.5, because that is what mlx-community had
  quantized on the day.

## 2026-09-22: Tier B, first pointer-head run on Qwen3-4B features

Setup: 3,000 records sampled from Open-Jev, tasksource-jev, and the jev-bench train
splits (300 per config), filtered to at most 160 options. Six whole sources held out
(civil_comments, fever_evidence, helpsteer2_helpfulness, stsb, yelp5, and one
tasksource task), giving 2,651 training records with one shuffled twin per choice
record (4,076 presentations) and 349 held-out records. Features: answer-position and
option-line hidden states, projected 2560 to 512 with a fixed seeded matrix and
RMS-normalized. Head: rank-64 pointer residual, zero-initialized query, per-kind scale,
noul bias. Loss: cross-entropy plus Brier plus 0.5 permutation-KL. AdamW 1e-3, gradient
clip 1.0, ten epochs, checkpoint selected by held-out Brier (epoch 9). Training takes
about four seconds; feature extraction took eight minutes.

Test shard: 1,100 rows sampled across all 22 jev-bench test configs, about 50 each.
These are raw head outputs versus raw backbone outputs, with no per-config calibration
on either side. Artifacts: `evals/results/tier-b-qwen3-4b.head.npz`,
`tier-b-qwen3-4b.history.json`, `tier-b-qwen3-4b-test.json`. The projection matrix is
not committed; it is `moelars.train.features.projection(2560, 512, seed=0)`.

| scope | acc base | acc head | ECE base | ECE head | Brier base | Brier head |
|---|---|---|---|---|---|---|
| macro over 22 configs | 0.658 | **0.684** | 0.305 | **0.213** | 0.587 | **0.462** |
| micro over 1,100 rows | 0.661 | **0.689** | 0.290 | **0.174** | 0.584 | **0.455** |
| in-distribution shard (4,076) | 0.681 | 0.851 | 0.283 | 0.030 | 0.569 | 0.207 |
| held-out shard (349) | 0.593 | 0.613 | 0.324 | 0.244 | 0.690 | 0.573 |

Per source on the test shard, largest moves:

| config | base | head | held-out | note |
|---|---|---|---|---|
| measuring_hate_speech | 0.390 | 0.593 | | +20 |
| go_emotions | 0.326 | 0.457 | | +13 |
| civil_comments | 0.611 | 0.685 | yes | +7, Brier 0.758 to 0.373 |
| ledgar | 0.741 | 0.815 | | +7 |
| strategyqa_grounded | 0.768 | 0.839 | | +7 |
| sst5 | 0.509 | 0.436 | | -7 |
| mmlu | 0.683 | 0.634 | | -5 |
| helpsteer2_helpfulness | 0.308 | 0.269 | yes | -4, Brier worse too |

### Reading

- **The head helps, and most of the help is calibration.** ECE drops by a third and
  Brier by a fifth on the test shard with no per-config fitting at all, because the
  head learns a per-kind scale on the backbone's logits. Accuracy rises 2.6 points
  macro. In-distribution it rises 17 points, which is the usual gap between what a head
  memorizes and what it generalizes.
- **Held-out sources are roughly neutral on accuracy and better on Brier**, which is the
  outcome the residual design is meant to buy: no regression on question forms the head
  never saw, with the calibration gain carried over. On 349 rows the accuracy numbers are
  within noise of the baseline. This matches the jev-bench maintainers' finding that a
  residual head keeps in-distribution gains without a held-out regression.
- **Where it hurts**: sst5 and mmlu lose 5 to 7 points on about 50 rows each. sst5 is an
  ordinal task whose train rows the head did see, so this may be the score head
  over-weighting adjacent levels. mmlu is knowledge, which no head fixes.
- **Fair comparison**: see the next section, which runs the head through the same
  per-config calibration as Tier A.

### Bugs found on the way, both now tested

- Quantized output projections pack the weight; selecting rows without dequantizing
  gave 320-wide garbage.
- Padded option positions in the permutation-KL scatter added log-probabilities near
  minus a billion onto option zero. The loss started at two billion. Masking before the
  scatter fixed it; a unit test pins the zero-KL identity case with padding.

## 2026-09-22 (earlier): first three configs

Superseded by the full table above; kept in git history.

## 2026-09-22: Tier B fair comparison, head plus per-config calibration

Same 22 configs, same 200 test rows, same calibration procedure on the same 200
validation rows, with every pass routed through the first pointer head
(`evals/results/tier-b-qwen3-4b.head.npz`). Full accuracy, ECE, Brier, and latency
tables are in `evals/results/compare-qwen3-4b-head.md`; the run itself is
`evals/results/qwen3-4b-instruct-2507-4bit-head.{json,md}`.

| scope | n | acc 4B | acc 4B+head | ECE 4B | ECE 4B+head | Brier 4B | Brier 4B+head | Jev acc |
|---|---|---|---|---|---|---|---|---|
| all | 22 | 0.662 | **0.680** | 0.088 | **0.082** | 0.404 | **0.390** | 0.733 |
| choice | 9 | 0.657 | **0.676** | **0.084** | 0.097 | 0.386 | **0.374** | 0.770 |
| score | 6 | 0.466 | **0.492** | 0.116 | **0.090** | 0.645 | **0.613** | 0.503 |
| noul | 7 | 0.837 | **0.847** | 0.069 | **0.056** | 0.222 | **0.219** | 0.881 |
| sources the head trained on | 16 | 0.668 | **0.699** | | | 0.410 | **0.385** | |
| sources held out of training | 5 | **0.639** | 0.632 | | | **0.434** | 0.444 | |

Latency is unchanged: 138 ms per row without the head, 135 with it, because the hidden
states come out of the same forward pass and the head is a few matrix products in numpy.

Largest moves, calibrated accuracy:

| config | 4B | 4B+head | seen in training | note |
|---|---|---|---|---|
| measuring_hate_speech | 0.405 | 0.615 | yes | ahead of Jev's 0.527 |
| go_emotions | 0.245 | 0.395 | yes | ahead of Jev's 0.282, but ECE 0.05 to 0.16 |
| ledgar | 0.635 | 0.700 | yes | |
| mnli | 0.725 | 0.770 | yes | |
| strategyqa_closed | 0.570 | 0.605 | yes | |
| helpsteer2_helpfulness | 0.325 | 0.280 | held out | Brier worse too |
| chaosnli | 0.685 | 0.620 | borrows mnli calibrator | sharper mnli answers hurt the vote-distribution target |
| clinc150 | 0.725 | 0.700 | yes | 151-way; head was trained on 300 rows of it |
| sst5 | 0.475 | 0.450 | yes | |

### Reading

- **The honest gain is +1.8 macro accuracy**, not the +2.6 of the raw-versus-raw
  comparison, and most of the earlier calibration gain disappears once Tier A gets its
  own per-config fit. Brier still improves on 14 of 22 configs.
- **All of the gain is in-distribution.** On the 16 sources the head trained on it is
  +3.1 accuracy and Brier down 0.025; on the 5 held-out sources it is 0.7 down on
  accuracy and 0.01 worse on Brier, within noise for 200-row splits (one standard error
  is about 3.4 points at these accuracies) but not a gain. This is the residual-head
  promise kept at the minimum: no regression on new forms, and no free lunch either.
- **The head sharpens.** ECE gets worse on go_emotions, ledgar, mmlu, and the score
  tasks it did not see. The learned per-kind scale is fitted to the training mix; where
  a config's temperature disagrees with it, the per-config temperature has to undo it.
- **chaosnli is a trap.** It has no validation split, so it borrows mnli's calibrator;
  the head's sharper mnli makes chaosnli's soft targets look worse. A config that scores
  against vote distributions should get its own temperature or a flatter one.
- **Recommendation for serving**: the head is worth switching on for a deployment
  whose question forms resemble the training corpus (intent routing, sentiment,
  toxicity, NLI). For unknown forms, Tier A plus a per-config fit is as good and simpler.

### Not done in this batch

A second head trained on the full corpus, the Qwen3.5-9B backbone, and the Molar Triage
example numbers were queued and cancelled for a reboot. The head and a partial 9B run
are in the next two sections.

## 2026-09-22: Tier B head v2, full corpus, fair comparison

Same recipe as the first head, trained on the full clean corpus: 14,285 records
(`--limit 20000`, at most 160 options), the same five jev-bench sources plus one
tasksource task held out, giving 12,685 training records (19,637 presentations with
shuffled twins) and 1,600 held-out records. Feature extraction took 33 minutes on the
4B; training takes under a second per epoch. The checkpoint was selected at epoch 4
on held-out Brier; later epochs kept improving training accuracy (0.83 by epoch 10)
while held-out accuracy fell, so more epochs would not help. Artifacts:
`evals/results/tier-b-qwen3-4b-v2.head.npz`, `tier-b-qwen3-4b-v2.history.json`, and the
raw per-source test-shard report `tier-b-qwen3-4b-v2-test.json`; the projection is the
same seeded `projection(2560, 512, seed=0)` as the first head. Full tables for Tier A,
head v1, and head v2, all with per-config calibration: `evals/results/compare-qwen3-4b-head-v2.md`.

| scope | n | Tier A | head v1 | head v2 | Jev |
|---|---|---|---|---|---|
| accuracy, all | 22 | 0.662 | 0.680 | **0.689** | 0.733 |
| accuracy, choice | 9 | 0.657 | 0.676 | **0.701** | 0.770 |
| accuracy, score | 6 | 0.466 | **0.492** | 0.484 | 0.503 |
| accuracy, noul | 7 | 0.837 | 0.847 | **0.849** | 0.881 |
| accuracy, sources the head trained on | 16 | 0.668 | 0.699 | **0.708** | |
| accuracy, sources held out of training | 5 | **0.639** | 0.632 | 0.628 | |
| ECE, all | 22 | 0.088 | 0.082 | **0.073** | 0.113 |
| Brier, all | 22 | 0.404 | 0.390 | **0.372** | 0.349 |

### Reading

- **+2.7 macro accuracy over Tier A with calibration on both sides**, up from +1.8 for
  the first head. The gap to Jev is 4.4 points, down from 7.1.
- **The new gain is on high-K routing**: banking77 +4.0, clinc150 +4.0, massive +4.5
  over Tier A, with ECE on clinc150 halved (0.147 to 0.072). These were the largest
  non-knowledge gaps.
- **Still no gain on held-out sources.** The raw per-source report on the test shard
  showed civil_comments, a held-out source, up 18 points, but per-config calibration
  gives Tier A the same answer, so after calibration held-out sources are flat to
  slightly down (0.628 against 0.639, within one standard error). Everything this head
  buys is on question forms that the training corpus covers.
- **chaosnli recovered** to Tier A's 0.685 (head v1: 0.620). It is not the borrowed
  calibrator: mnli's fitted choice temperature is about the same under both heads (1.44
  and 1.34), so the difference is in the heads' chaosnli outputs. Not investigated
  further; on 200 rows 6.5 points is about two standard errors.
- **Score tasks are flat.** The ordinal configs are where the head does least; sst5,
  yelp5, and stsb move within noise.
- **Where the remaining 4.4 points are**, in macro points: closed-book knowledge (mmlu,
  strategyqa_closed, arc_challenge) 2.5; reading and NLI (mnli, stsb,
  strategyqa_grounded, paws, boolq, fever) 2.1; high-K routing (banking77, clinc150,
  massive, ledgar) 1.6; ordinal scores (sst5, yelp5, helpsteer2_helpfulness) 1.1;
  sms_spam 0.2; offset by 3.2 points of configs where moe-LARS is ahead. civil_comments
  is 1.8 of those 3.2 and is the majority baseline (see the Tier A caveats), so the
  headline flatters moe-LARS by about that much.
- **Latency is not reported for this run.** Another job shared the GPU while it ran;
  three configs came out five to eight times slower than head v1 on identical compute.
  Load time, peak memory, and warm latency are measured separately by
  `scripts/load_cost.py`.

## 2026-09-22: Qwen3.5-9B Tier A, partial (11 of 22 configs)

`mlx-community/Qwen3.5-9B-MLX-4bit`, same suite and calibration procedure. Stopped by
hand after 11 configs: it was not beating 4B plus head v2, at two to three times the
cost per row, and the GPU was needed. The suite writes after every config, so
`evals/results/qwen3-5-9b-mlx-4bit.{json,md}` hold the 11 finished configs; their macro
rows cover those 11 only. Latency was measured with another job on the GPU and is only a
rough ratio.

| config | 4B | 4B + head v2 | 9B | Jev | 9B ms/row | 4B ms/row |
|---|---|---|---|---|---|---|
| banking77 | 0.665 | **0.705** | 0.695 | 0.796 | 743 | 217 |
| boolq | **0.890** | **0.890** | 0.870 | 0.917 | 1052 | 107 |
| sst5 | **0.475** | 0.470 | **0.475** | 0.565 | 186 | 89 |
| clinc150 | 0.725 | 0.765 | **0.780** | 0.893 | 716 | 369 |
| massive | 0.675 | **0.720** | 0.705 | 0.808 | 392 | 216 |
| ledgar | 0.635 | **0.700** | 0.665 | 0.751 | 420 | 226 |
| go_emotions | 0.245 | **0.430** | 0.230 | 0.282 | 362 | 107 |
| mmlu | 0.670 | 0.640 | **0.710** | 0.923 | 184 | 105 |
| arc_challenge | 0.890 | 0.900 | **0.910** | 0.979 | 157 | 96 |
| mnli | 0.725 | 0.765 | **0.805** | 0.883 | 172 | 107 |
| chaosnli | **0.685** | **0.685** | 0.670 | 0.615 | 169 | 106 |
| mean of these 11 | 0.662 | **0.697** | 0.683 | 0.765 | | |

### Reading

- **Dense scale is the expensive lever.** On these 11 configs the 9B adds 2.1 points
  over the 4B and 4B plus head v2 adds 3.5. The jev-bench maintainers' zero-shot
  Qwen3.5-9B lands at 0.689 macro, which is where our 4B plus head already is.
- **The 9B helps where knowledge and reading matter** (mmlu +4, mnli +8, clinc150 +5.5)
  and not on label-prior tasks (go_emotions, chaosnli), which is the opposite profile
  to the head. The two levers are complementary.
- **The 9B's raw logits are already close to calibrated**: fitted choice temperatures
  of 1.1 to 1.3 against 4 to 5 for the 4B. On 200 validation rows the fit sometimes
  makes test ECE slightly worse (clinc150 0.062 raw to 0.122).
- **Next levers**, in order: LoRA on the 4B plus a head (the jev-bench maintainers'
  LoRA plus residual head on the same 4B reaches 0.747, above Jev), a confidence
  cascade to a larger model for knowledge questions, and a mixture-of-experts backbone
  (Qwen3-30B-A3B, about 3B active parameters) as the cascade target or default.

### Bug found on the way, now tested

Qwen3.5's Gated DeltaNet layers keep their state in an `ArraysCache`, whose `.state`
hands out the internal list and whose setter adopts the list it is given; the layers
then write through `cache[i] = ...`. Restoring a prefix snapshot for one row therefore
changed the snapshot for the next. Snapshots and restores now copy the list, and a unit
test with the real mlx-lm cache classes pins it.

## 2026-09-22: LoRA on the 4B, LoRA plus head, and Qwen3-30B-A3B Tier A

Three runs from `scripts/queue_lora.sh`, all with per-config calibration:

- **LoRA** (`moelars.train.lora`): rank 8, scale 20, every layer, lr 2e-5 with warmup and
  cosine decay, one epoch over the same 14,285-record corpus and held-out split as head
  v2 (1,095 steps, 113 minutes, about 27 GB peak). Loss is cross-entropy plus Brier on
  the K label logits at the answer position, with options reshuffled per presentation.
  Held-out Brier went 0.707 to 0.479 and held-out accuracy 0.590 to 0.606; the final step
  was the best checkpoint. Training history and adapter config:
  `evals/results/lora-4b.{history,adapter_config}.json`. The adapter itself (66 MB) is
  not committed.
- **LoRA plus head**: features re-extracted through the adapter, a pointer head trained
  on them with the head v2 recipe.
- **Qwen3-30B-A3B-Instruct-2507 4-bit** zero-shot (MoE, about 3B active parameters,
  18 GB peak).

| scope | n | 4B Tier A | 4B + head v2 | 4B + LoRA | 4B + LoRA + head | 30B-A3B | Jev |
|---|---|---|---|---|---|---|---|
| accuracy, all | 22 | 0.662 | 0.689 | **0.731** | 0.729 | 0.680 | 0.733 |
| accuracy, choice | 9 | 0.657 | 0.701 | **0.757** | **0.757** | 0.706 | 0.770 |
| accuracy, score | 6 | 0.466 | 0.484 | **0.520** | 0.513 | 0.416 | 0.503 |
| accuracy, noul | 7 | 0.837 | 0.849 | **0.878** | 0.877 | 0.873 | 0.881 |
| accuracy, sources trained on | 16 | 0.668 | 0.708 | **0.775** | **0.775** | 0.691 | |
| accuracy, sources held out | 5 | 0.639 | 0.628 | 0.607 | 0.605 | **0.649** | |
| held out, without stsb | 4 | 0.700 | 0.690 | 0.700 | 0.701 | **0.705** | |
| ECE, all | 22 | 0.088 | **0.073** | 0.077 | **0.073** | 0.079 | 0.113 |
| Brier, all | 22 | 0.404 | 0.372 | **0.317** | 0.318 | 0.372 | 0.349 |

Per-config tables: `evals/results/qwen3-4b-instruct-2507-4bit-{lora,lora-head}.md` and
`qwen3-30b-a3b-instruct-2507-4bit.md`.

### Reading

- **LoRA ties Jev on macro accuracy (0.731 against 0.733) and beats it on Brier (0.317
  against 0.349) and ECE (0.077 against 0.113).** It is +4.2 over head v2 and +6.9 over
  Tier A. The civil_comments caveat above still applies: that config is the majority
  baseline and is worth about 1.8 macro points of moe-LARS's lead where it leads.
- **The gain is on question forms the corpus covers**: seen sources 0.775 against 0.668
  for Tier A. Largest moves: measuring_hate_speech +36, go_emotions +23, clinc150 +16.5,
  massive +15, banking77 +12.5, ledgar +11, mnli +10.
- **Held-out sources are flat except stsb.** Four of the five held-out sources average
  0.700 under LoRA, the same as Tier A. stsb falls from 0.395 to 0.235 and its fitted
  temperature from 6.9 to 2.6: the adapter makes the model confident on a 6-level
  similarity scale it never saw, in the wrong places. This is the one regression that
  matters before LoRA becomes a default, and the reason the next training change is on
  score tasks (ordinal-aware loss, more score sources), not more epochs.
- **chaosnli falls 0.685 to 0.640.** It borrows mnli's calibrator, and mnli moved most
  under the adapter, so some of this may be the borrowed temperature. Not separated yet.
- **A head on top of LoRA adds nothing in total** (0.729). With the adapter, training
  accuracy on the corpus is already 0.90, and no head epoch beat the adapter alone on
  held-out Brier (0.479 alone, best epoch 0.485). The head's selection loop did not
  compare against no head at all and saved that slightly worse epoch; it now does
  (`moelars.train.residual`, tested). By config the head still helps high-K routing
  (massive +3.0, ledgar +4.5, mnli +2.0) and hurts elsewhere (mmlu -4.5, chaosnli -3.0),
  which suggests gating it on K. That rule was read off test, so it is a hypothesis to
  check on held-out data, not a result.
- **The 30B-A3B is 0.680 zero-shot**: +1.8 over the 4B's Tier A, 5.1 behind the adapted
  4B, and about twice as slow per row as the 4B Tier A (266 against 138 ms mean, shared
  GPU). It is better than any 4B variant on held-out sources (0.649), on knowledge
  (mmlu 0.780 against 0.690 for LoRA, arc_challenge 0.950 against 0.935), and on stsb
  (0.425 against 0.235). Those are the adapted 4B's weak spots, so the two are
  complementary, which is the case for an `escalate_to` cascade with the 4B plus LoRA as
  the primary.
- **30B helpsteer2_verbosity is 0.110.** Calibration cannot help (0.110 raw and
  calibrated, raw ECE 0.816): the model puts nearly all its mass on one end of the
  scale. At a plausible 0.6 it would be about 2 macro points higher. Not investigated.
- **Latency** for these runs was measured with another job on the GPU. LoRA rows were
  also slower than Tier A because the adapter was applied unfused; see the fused check
  below.

### Fusing the adapter

`mlx_lm.fuse` two ways, each run on 7 configs with the suite's own per-config
calibration and compared with the unfused adapter above (accuracy, calibrated):

| config | 4B Tier A | unfused LoRA | fused, bf16 | fused, 4-bit |
|---|---|---|---|---|
| banking77 | 0.665 | 0.790 | 0.785 | 0.705 |
| mnli | 0.725 | 0.825 | 0.820 | 0.745 |
| stsb | 0.395 | 0.235 | 0.235 | 0.340 |
| sst5 | 0.475 | 0.510 | 0.510 | 0.495 |
| boolq | 0.890 | 0.910 | 0.910 | 0.895 |
| mmlu | 0.670 | 0.690 | 0.700 | 0.670 |
| helpsteer2_verbosity | 0.585 | 0.675 | 0.675 | 0.665 |
| mean of these 7 | 0.629 | 0.662 | 0.662 | 0.645 |

- **bf16 fusion is faithful**: same mean accuracy, Brier within 0.001 on every config,
  and 25 to 40 percent less time per row than the unfused adapter on the same shared GPU
  (for example mnli 116 against 150 ms, stsb 115 against 195). The cost is size: 7.5 GB
  on disk against 2.1 GB for the 4-bit base plus a 66 MB adapter.
- **4-bit fusion loses about half the adapter.** Re-quantizing after adding the update
  rounds most of it away: banking77 falls back 8.5 of the adapter's 12.5 points, and stsb
  recovers toward Tier A for the same reason. Do not serve a re-quantized fused model;
  training the adapter against quantization (or DWQ-style distillation into the 4-bit
  weights) would be needed first.
