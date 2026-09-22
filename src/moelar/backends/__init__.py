"""Model backends. Each one prefills a shared prefix and reads next-token logits per suffix."""

from __future__ import annotations

from moelar.backends.base import Backend


def load_backend(name: str, model: str | None = None, template: str | None = None,
                 adapter: str | None = None) -> Backend:
    if adapter and name != "mlx":
        raise ValueError("LoRA adapters are only supported on the mlx backend")
    if name == "mock":
        from moelar.backends.mock import MockBackend

        return MockBackend()
    if name == "mlx":
        from moelar.backends.mlx import MLXBackend

        if not model:
            raise ValueError("--model is required for the mlx backend")
        return MLXBackend(model, template=template, adapter=adapter)
    if name == "llamacpp":
        from moelar.backends.llamacpp import LlamaCppBackend

        if not model:
            raise ValueError("--model is required for the llamacpp backend")
        return LlamaCppBackend(model, template=template or "chatml")
    raise ValueError(f"unknown backend {name!r}; choose mock, mlx, or llamacpp")


__all__ = ["Backend", "load_backend"]
