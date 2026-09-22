# Tier B: residual decision head

## Recipe

The backbone stays frozen. For each row, the engine already computes label logits
`z` (one per option) by reading the next-token distribution after `Answer:`. Tier B
adds a small head that produces a correction `r` and a learned scale, and serves

    z' = s * z + r,     r = head(h, E_labels)

where `h` is the backbone's last hidden state at the answer position and `E_labels`
are the output-embedding rows of the label tokens. In practice the head is a pointer:
it scores each option's line-end hidden state against the answer-position hidden
state. Both are projected with a fixed seeded matrix and RMS-normalized, identically
at training and serving time, because raw final-layer hidden states have norms in the
hundreds and an unnormalized head diverges on the first step. The head's query
projection is zero-initialized so training starts exactly at the Tier A model. This is the
"residual on the model's own scorer" design that jev-bench's maintainers found keeps
the in-distribution gain without regressing on sources the head never saw.

Losses, summed:

1. cross-entropy against the target distribution (soft labels when available)
2. Brier against the same target, which is what calibration is scored on
3. permutation-KL: each choice row is also presented with shuffled option order, and
   the two realigned distributions are pulled together
4. complement consistency: for noul pairs generated as negations, P(yes) of one plus
   P(yes) of the other is pulled toward 1

Sources are split by whole source, never by row, so the held-out number measures
generalization to question forms the head never saw.

## Data

    uv run python -m moelar.train.build --out data/train

pulls Open-Jev (CC0), tasksource-jev, and the jev-bench train and validation splits,
converts them to `Record` JSONL, and writes a manifest with per-source counts.

No data labeled by Jev is used. See `DESIGN.md` section 8.

## Commands

```bash
# 1. corpus (CPU, downloads)
uv run python -m moelar.train.build --out data/train --max-per-source 4000 --jev-bench-per-config 300

# 2. features from the frozen backbone (GPU). Whole sources are held out.
uv run python -m moelar.train.extract --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --records data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl \
    --test-records evals/data/*.test.jsonl --limit 3000 --holdout-fraction 0.2 --out data/features

# 3. train the head on cached features (seconds)
uv run python -m moelar.train.residual --train data/features/train.npz --heldout data/features/heldout.npz \
    --out checkpoints/pointer_head.npz

# 4. serve with it
uv run moelar serve --backend mlx --model <model> --head checkpoints/pointer_head.npz \
    --projection data/features/projection.npy
```

## Status

First real run done on Qwen3-4B features, see `evals/RESULTS.md`: +2.6 macro accuracy,
ECE down a third, Brier down a fifth on the jev-bench test shard with no per-config
calibration; held-out sources neutral on accuracy and better on Brier. Checkpoint
selection is by held-out Brier. `python -m moelar.train.report` gives the per-source
table. Next: head plus per-config temperature against the Tier A table, more training
records, and a 9B backbone.

Known gaps: the head is per-letter-position-blind by design (it scores option text
through the option-line hidden state, not the letter), so it cannot fix letter bias in
the backbone's own `z`; the learned scale can only shrink it. Complement-consistency
loss is described above but not yet implemented, since no source in the corpus
supplies negated noul pairs; generate them in `build` first.
