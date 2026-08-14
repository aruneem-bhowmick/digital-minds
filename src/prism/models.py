"""Model + SAE loading (REQ-1, REQ-2).

REQ-1 / ADR-0010: the SAE dependency resolves to an existing checkpoint, not
a trained-from-scratch fallback. ``load_model_and_sae`` loads Pythia-70m-deduped
through TransformerLens and pairs it with the exact residual-stream SAE the
frame-theoretic identifiability audit already scored (Hugging Face
``ghidav/pythia-70m-deduped-sae``, layer 4, ``blocks.4.hook_resid_pre``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from sae_lens import SAE
from transformer_lens import HookedTransformer


@dataclass
class LoadedPrismModel:
    """A base model paired with the SAE whose dictionary was scored by the identifiability audit."""

    model: HookedTransformer
    sae: SAE
    hook_name: str


def load_model_and_sae(config: dict[str, Any], device: str = "cpu") -> LoadedPrismModel:
    """Load the base model and its matching SAE per ADR-0001 / ADR-0002 / ADR-0010.

    ``config`` is the parsed contents of ``configs/experiment.yaml`` (or an
    equivalent dict with ``model`` and ``sae`` sections). Raises if the
    configured hook name doesn't exist on the loaded model or doesn't match
    the hook the SAE was trained against -- a wrong-layer config error
    should surface here, not several modules downstream during injection.
    """
    model_name = config["model"]["name"]
    model = HookedTransformer.from_pretrained(model_name, device=device)

    sae_cfg = config["sae"]
    hook_name = sae_cfg["hook_name"]

    checkpoint_dir = _download_sae_checkpoint(sae_cfg)
    sae = SAE.load_from_disk(checkpoint_dir, device=device)

    _validate_pairing(model, sae, hook_name, model_name=model_name)

    return LoadedPrismModel(model=model, sae=sae, hook_name=hook_name)


def _validate_pairing(model: Any, sae: Any, hook_name: str, *, model_name: str = "") -> None:
    """Check that a model/SAE/hook-name combination is actually consistent.

    Split out from ``load_model_and_sae`` so the failure paths (wrong-layer
    or mismatched-checkpoint config, the class of bug CLAUDE.md calls out as
    the one most likely to invalidate every downstream result) are testable
    without downloading a model or SAE checkpoint.
    """
    if hook_name not in model.hook_dict:
        raise ValueError(
            f"configured hook_name {hook_name!r} does not exist on {model_name!r}; "
            "check configs/experiment.yaml against the model's actual hook points"
        )
    if sae.cfg.metadata.hook_name != hook_name:
        raise ValueError(
            f"configured hook_name {hook_name!r} does not match the hook this SAE "
            f"was trained against ({sae.cfg.metadata.hook_name!r})"
        )
    if sae.cfg.d_in != model.cfg.d_model:
        raise ValueError(
            f"SAE d_in ({sae.cfg.d_in}) does not match {model_name!r}'s d_model "
            f"({model.cfg.d_model}) -- wrong model/SAE pairing"
        )


def _download_sae_checkpoint(sae_cfg: dict[str, Any]) -> Path:
    """Download the audited SAE checkpoint at its pinned revision and verify its checksum.

    Per CLAUDE.md's reproducibility rule: the revision is pinned to a specific
    commit (never ``main``), and the downloaded weights are checked against the
    checksum recorded in ``configs/experiment.yaml`` before use.
    """
    repo_id = sae_cfg["checkpoint_repo"]
    revision = sae_cfg["checkpoint_revision"]
    subfolder = sae_cfg["checkpoint_subfolder"]
    expected_sha256 = sae_cfg["checkpoint_sha256"]

    weights_path = None
    for filename in ("cfg.json", "sae_weights.safetensors"):
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/{filename}",
            revision=revision,
        )
        if filename == "sae_weights.safetensors":
            weights_path = Path(downloaded)

    assert weights_path is not None
    _verify_sha256(weights_path, expected_sha256)
    return weights_path.parent


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {digest} -- "
            "the downloaded checkpoint does not match the one the identifiability audit scored"
        )

