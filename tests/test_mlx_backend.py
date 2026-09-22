"""MLX backend plumbing that does not need a real checkpoint. Skipped where MLX is unavailable."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from moelar.backends.mlx import MLXBackend  # noqa: E402


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text.strip()] if len(text.strip()) == 1 else [1, 2]


def _backend(model) -> MLXBackend:
    backend = object.__new__(MLXBackend)
    backend._mx = mx
    backend.model = model
    backend._text = getattr(model, "language_model", model)
    backend.tokenizer = _Tokenizer()
    backend._label_cache = {}
    return backend


def test_vision_language_wrapper_resolves_text_stack():
    """Qwen3.5-style checkpoints keep the text model under `language_model`."""
    weight = mx.arange(128 * 8, dtype=mx.float32).reshape(128, 8)
    text = SimpleNamespace(args=SimpleNamespace(hidden_size=8, tie_word_embeddings=False),
                           model=SimpleNamespace(embed_tokens=None), lm_head=SimpleNamespace(weight=weight))
    backend = _backend(SimpleNamespace(language_model=text))
    assert backend.hidden_size == 8
    rows = backend.label_rows(["A", "B"])
    assert rows.shape == (2, 8)
    assert np.allclose(rows, np.asarray(weight)[[ord("A"), ord("B")]])


def test_plain_model_with_tied_embeddings():
    weight = mx.arange(128 * 4, dtype=mx.float32).reshape(128, 4)
    model = SimpleNamespace(args=SimpleNamespace(hidden_size=4, tie_word_embeddings=True),
                            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)))
    backend = _backend(model)
    assert backend.hidden_size == 4
    assert np.allclose(backend.label_rows(["Z"]), np.asarray(weight)[[ord("Z")]])


def test_label_rows_rejects_multi_token_labels():
    backend = _backend(SimpleNamespace(args=SimpleNamespace(hidden_size=4, tie_word_embeddings=True),
                                       model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mx.zeros((128, 4))))))
    with pytest.raises(ValueError):
        backend.label_rows(["AB"])
