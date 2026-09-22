# Tier B: residual decision head

## Recipe

The backbone stays frozen. For each row, the engine already computes label logits
`z` (one per option) by reading the next-token distribution after `Answer:`. Tier B
adds a small head that produces a correction `r` and a learned scale, and serves

    z' = s * z + r,     r = head(h, E_labels)

where `h` is the backbone's last hidden state at the answer position and `E_labels`
are the output-embedding rows of the label tokens. The head's last layer is
zero-initialized so training starts exactly at the Tier A model. This is the
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

## Status

Data layer written. Feature extraction (hidden state at the answer position) and the
MLX training loop are next. Training runs on the same Apple Silicon box as serving.
