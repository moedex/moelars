"""Named serving configurations: `moelars serve --preset 30b`.

A preset names the backend, the base model, one LoRA adapter or several (served as an
ensemble, `moelars.engine.EnsembleEngine`), and the calibrator that belongs with each.
Adapters and calibrators live on the Hugging Face Hub and are downloaded on first use; a
local directory or file with the same argument is used as is.
The published adapters are trained on the commercially licensed corpus (DESIGN.md 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

M30 = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
CALIBRATION_FILE = "moelars-calibration.json"


@dataclass(frozen=True)
class Preset:
    backend: str
    model: str
    adapters: tuple[str, ...]
    description: str

    @property
    def calibrations(self) -> list[str]:
        """Each adapter's pooled calibrator is published inside that adapter's repository."""
        return [f"{adapter}/{CALIBRATION_FILE}" for adapter in self.adapters]


PRESETS: dict[str, Preset] = {
    "30b": Preset("mlx", M30, ("moedex/moelars-qwen3-30b-a3b-lora-attn",),
                  "Qwen3-30B-A3B 4-bit with the attention LoRA and its pooled calibrator"),
}


def _is_hub_id(value: str) -> bool:
    return not Path(value).exists() and value.count("/") == 1 and not value.startswith((".", "/", "~"))


def resolve_adapter(adapter: str | None) -> str | None:
    """A local adapter directory, or a Hub repository ID downloaded to the local cache."""
    if adapter is None or not _is_hub_id(adapter):
        return adapter
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=adapter)


def resolve_calibration(calibration: str | None) -> str | None:
    """A local calibrator file, or `<org>/<repo>/<file>` downloaded from a Hub repository."""
    if calibration is None or Path(calibration).exists():
        return calibration
    repo, _, filename = calibration.rpartition("/")
    if repo.count("/") != 1 or not filename:
        return calibration
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)
