"""MLX backend for Apple Silicon.

Prefills the shared prefix once into a KV cache, snapshots it, and for each suffix
restores the snapshot and runs the suffix to read the last-position logits.

Requires `pip install moelar[mlx]`. Untested paths are guarded: if snapshot or restore
fails on a given cache type, the backend falls back to a full prefill per row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from moelar.backends.base import Backend
from moelar.render import TEMPLATES, TemplateFn


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_path: str, template: str | None = None) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as error:  # pragma: no cover - depends on platform
            raise ImportError("install with `pip install moelar[mlx]` on Apple Silicon") from error

        self._mx = mx
        self._make_cache = make_prompt_cache
        self.model, self.tokenizer = load(model_path)
        self.model_name = model_path
        self._template: TemplateFn
        if template and template in TEMPLATES:
            self._template = TEMPLATES[template]
        elif getattr(self.tokenizer, "chat_template", None):
            self._template = self._hf_template
        else:
            self._template = TEMPLATES["chatml"]
        self._label_cache: dict[str, int | None] = {}

    # ----------------------------------------------------------------- template + tokens

    def _hf_template(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def template(self) -> TemplateFn:
        return self._template

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _label_id(self, label: str) -> int | None:
        if label not in self._label_cache:
            ids = self._encode(" " + label)
            self._label_cache[label] = ids[0] if len(ids) == 1 else None
        return self._label_cache[label]

    def is_single_token(self, label: str) -> bool:
        return self._label_id(label) is not None

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    # ----------------------------------------------------------------- inference

    def _forward(self, ids: list[int], cache: Any) -> np.ndarray:
        mx = self._mx
        logits = self.model(mx.array(ids)[None], cache=cache)
        last = logits[0, -1, :]
        mx.eval(last)
        return np.asarray(last.astype(mx.float32))

    def _snapshot(self, cache: Any) -> list[Any] | None:
        try:
            return [(c.state, getattr(c, "meta_state", None)) for c in cache]
        except Exception:  # pragma: no cover - cache type without state API
            return None

    def _restore(self, snapshot: list[Any]) -> Any:
        cache = self._make_cache(self.model)
        for c, (state, meta) in zip(cache, snapshot, strict=True):
            c.state = state
            if meta is not None and hasattr(c, "meta_state"):
                c.meta_state = meta
        return cache

    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        prefix_ids = self._encode(prefix)
        prefix_cache = self._make_cache(self.model)
        self._forward(prefix_ids, prefix_cache)
        snapshot = self._snapshot(prefix_cache)

        results: list[np.ndarray] = []
        for suffix, row_labels in zip(suffixes, labels, strict=True):
            full_ids = self._encode(prefix + suffix)
            shares_prefix = full_ids[: len(prefix_ids)] == prefix_ids
            if snapshot is not None and shares_prefix:
                cache = self._restore(snapshot)
                vocab_logits = self._forward(full_ids[len(prefix_ids) :], cache)
            else:
                vocab_logits = self._forward(full_ids, self._make_cache(self.model))
            ids = [self._label_id(label) for label in row_labels]
            if any(i is None for i in ids):
                raise ValueError(f"labels not single-token for this tokenizer: {row_labels}")
            results.append(vocab_logits[np.asarray(ids)].astype(np.float64))
        return results
