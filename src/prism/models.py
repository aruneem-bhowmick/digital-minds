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
    model_revision = config["model"]["checkpoint_revision"]
    model = HookedTransformer.from_pretrained(model_name, revision=model_revision, device=device)

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
    commit (never ``main``), and every downloaded file is checked against the
    checksum recorded in ``configs/experiment.yaml`` before use -- including
    ``cfg.json``, which determines the hook name and dimensions
    ``_validate_pairing`` trusts, not just the weights tensor.
    """
    repo_id = sae_cfg["checkpoint_repo"]
    revision = sae_cfg["checkpoint_revision"]
    subfolder = sae_cfg["checkpoint_subfolder"]
    expected_sha256_by_filename = {
        "cfg.json": sae_cfg["checkpoint_cfg_sha256"],
        "sae_weights.safetensors": sae_cfg["checkpoint_sha256"],
    }

    weights_path = None
    for filename, expected_sha256 in expected_sha256_by_filename.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                revision=revision,
            )
        )
        _verify_sha256(downloaded, expected_sha256)
        if filename == "sae_weights.safetensors":
            weights_path = downloaded

    assert weights_path is not None
    return weights_path.parent


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {digest} -- "
            "the downloaded checkpoint does not match the one the identifiability audit scored"
        )


def validate_reconstruction(loaded: LoadedPrismModel, prompts: list[str]) -> dict[str, Any]:
    """Encode/decode real residual-stream activations through the SAE and report
    fraction of variance explained, per REQ-1's definition of done: reconstruction
    quality is reported, not assumed, regardless of which ADR-0002 branch was taken.
    """
    import torch

    activations: list[torch.Tensor] = []

    def _capture(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
        activations.append(act.detach())
        return act

    for prompt in prompts:
        tokens = loaded.model.to_tokens(prompt)
        loaded.model.run_with_hooks(tokens, fwd_hooks=[(loaded.hook_name, _capture)])

    acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
    reconstructed = loaded.sae.decode(loaded.sae.encode(acts))

    total_sum_sq = (acts - acts.mean(dim=0, keepdim=True)).pow(2).sum()
    if total_sum_sq == 0:
        raise ValueError(
            f"activations have zero variance across {acts.shape[0]} token(s); "
            "fraction_variance_explained is undefined for this input -- pass more or longer prompts"
        )
    residual_sum_sq = (acts - reconstructed).pow(2).sum()
    fraction_variance_explained = 1.0 - (residual_sum_sq / total_sum_sq).item()

    return {
        "n_tokens": int(acts.shape[0]),
        "fraction_variance_explained": fraction_variance_explained,
    }
