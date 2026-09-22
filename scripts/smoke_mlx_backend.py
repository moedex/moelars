"""Prefix-cache and feature-path consistency check for an MLX model. Exits non-zero on mismatch.

Run this before benchmarking a new architecture: it checks that restoring the prefix cache
gives the same label logits as a full prefill, that the feature path's dequantized label
rows reproduce the full forward pass, and that repeated rows are deterministic.

    uv run python scripts/smoke_mlx_backend.py mlx-community/Qwen3.5-9B-MLX-4bit
"""

import sys

import numpy as np

from moelar.backends.mlx import MLXBackend
from moelar.render import compose_prompt, render_choice, render_content
from moelar.schema import ChoiceQuestion


def main() -> int:
    backend = MLXBackend(sys.argv[1])
    print("hidden_size", backend.hidden_size)
    question = ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={"billing": "Payments, payouts", "technical": "Bugs, outages", "sales": "Pricing, demos"},
    )
    labels = ("A", "B", "C")
    body, _ = render_choice(question, list(labels), [0, 1, 2])
    state = render_content("Help! My payouts have been failing for 3 days.")
    prefix, suffix = compose_prompt(backend.template(), state, body)
    print("prompt tail:", repr((prefix + suffix)[-80:]))
    z_cached = backend.label_logits(prefix, [suffix, suffix], [labels] * 2)
    z_full = backend.label_logits("", [prefix + suffix], [labels])
    z_feat, h_ans, h_opt = backend.label_logits_with_features(prefix, [suffix], [labels])[0]
    print("cached", np.round(z_cached[0], 3), "full", np.round(z_full[0], 3), "features", np.round(z_feat, 3))
    print("h_ans norm", round(float(np.linalg.norm(h_ans)), 1), "h_opt", h_opt.shape)
    cached_vs_full = float(np.abs(z_cached[0] - z_full[0]).max())
    cached_vs_feat = float(np.abs(z_cached[0] - z_feat).max())
    repeat = float(np.abs(z_cached[0] - z_cached[1]).max())
    print("max diff cached-vs-full", cached_vs_full, "cached-vs-features", cached_vs_feat, "repeat", repeat)
    ok = cached_vs_full < 0.15 and cached_vs_feat < 0.15 and repeat < 1e-3 and int(np.argmax(z_cached[0])) == 0
    print("SMOKE", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
