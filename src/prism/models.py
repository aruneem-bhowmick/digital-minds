"""Model + SAE loading (REQ-1, REQ-2, REQ-11).

REQ-1 / ADR-0010: the SAE dependency resolves to an existing checkpoint, not
a trained-from-scratch fallback. ``load_model_and_sae`` loads Pythia-70m-deduped
through TransformerLens and pairs it with the exact residual-stream SAE the
frame-theoretic identifiability audit already scored (Hugging Face
``ghidav/pythia-70m-deduped-sae``, layer 4, ``blocks.4.hook_resid_pre``).

REQ-11: the same function also loads Gemma-2-2b paired with a Gemma Scope
residual-stream SAE, selected by ``config["sae"]["loader"]`` rather than
inferred from other fields -- Gemma Scope's ``.npz`` checkpoints need a
different download-and-construct path than Pythia's safetensors-plus-cfg.json
layout, so this is a second explicit branch, not a special case threaded
through the existing one. Everything downstream of ``LoadedPrismModel``
(``_validate_pairing``, ``measure_activation_frequencies``, ``decoder_norms``,
``top_activating_snippets``, ``validate_reconstruction``) is unmodified and
works against either model, since both loading paths return the same
``sae_lens.SAE`` interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from transformer_lens import HookedTransformer

if TYPE_CHECKING:
    import torch

# Fixed corpus for REQ-1's reconstruction-quality check (validate_reconstruction).
# Tokenizes to exactly 120 tokens under this model's tokenizer -- kept as one
# canonical list so the test and any future run reference the same corpus,
# rather than each defining its own ad hoc prompts.
RECONSTRUCTION_VALIDATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, astronauts landed on the surface of the Moon for the first time.",
    "She opened the door and found a small, curious cat sitting on the mat.",
    "The stock market fell sharply after the announcement of new tariffs.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "He picked up the phone, dialed the number, and waited for an answer.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and three eggs.",
    "Researchers published a new study on the effects of sleep deprivation on memory.",
]


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

    ``config["sae"]["loader"]`` picks which SAE-loading path runs
    (``"hf_safetensors"``, the default, for Pythia's checkpoint layout;
    ``"gemma_scope_npz"`` for REQ-11's Gemma Scope checkpoint). Checked
    before touching the network, so an unrecognized value fails immediately
    rather than after downloading the base model.
    """
    sae_cfg = config["sae"]
    hook_name = sae_cfg["hook_name"]
    loader = sae_cfg.get("loader", "hf_safetensors")
    if loader not in ("hf_safetensors", "gemma_scope_npz"):
        raise ValueError(
            f"unrecognized sae.loader {loader!r}; expected 'hf_safetensors' or 'gemma_scope_npz'"
        )

    model_name = config["model"]["name"]
    model_revision = config["model"]["checkpoint_revision"]
    model = HookedTransformer.from_pretrained(model_name, revision=model_revision, device=device)

    if loader == "hf_safetensors":
        checkpoint_dir = _download_sae_checkpoint(sae_cfg)
        sae = SAE.load_from_disk(checkpoint_dir, device=device)
    else:
        sae = _load_gemma_scope_sae(sae_cfg, device=device)

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


def _load_gemma_scope_sae(sae_cfg: dict[str, Any], device: str) -> SAE:
    """Download, checksum-verify, and load a Gemma Scope SAE (REQ-11).

    Gemma Scope ships ``.npz`` checkpoints -- a different format than the
    safetensors-plus-``cfg.json`` layout ``_download_sae_checkpoint`` reads
    for the Pythia checkpoint, so this is a second loading path rather than
    a branch inside that one.

    ``sae_lens.SAE.from_pretrained()`` has no ``revision`` parameter at
    all -- it resolves ``release``/``sae_id`` through its own registry with
    no way for a caller to pin which commit of ``checkpoint_repo`` that
    resolves to. Downloading and checksumming the file ourselves first
    (matching ``_download_sae_checkpoint``'s pattern) only proves the
    checksum passes on *a* download; it doesn't prove ``from_pretrained()``
    actually loaded those same bytes. So this also compares the two
    decoder matrices directly after loading, and raises if they don't
    match -- a real, enforced guarantee instead of an assumption the
    docstring used to state without checking it.
    """
    import numpy as np
    import torch

    downloaded = Path(
        hf_hub_download(
            repo_id=sae_cfg["checkpoint_repo"],
            filename=sae_cfg["checkpoint_filename"],
            revision=sae_cfg["checkpoint_revision"],
        )
    )
    _verify_sha256(downloaded, sae_cfg["checkpoint_sha256"])
    with np.load(downloaded) as archive:
        checksummed_w_dec = archive["W_dec"]

    sae = SAE.from_pretrained(
        release=sae_cfg["sae_lens_release"],
        sae_id=sae_cfg["sae_lens_sae_id"],
        device=device,
    )
    loaded_w_dec = sae.W_dec.detach().to(torch.float32).cpu().numpy()
    if not np.allclose(loaded_w_dec, checksummed_w_dec.astype(np.float32), atol=1e-6):
        raise ValueError(
            f"SAE.from_pretrained(release={sae_cfg['sae_lens_release']!r}, "
            f"sae_id={sae_cfg['sae_lens_sae_id']!r}) loaded a decoder matrix that does "
            f"not match the checksum-verified {downloaded} -- sae_lens resolved this "
            "release/sae_id to different weights than the pinned checkpoint_revision"
        )
    return sae


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

    Excludes each prompt's first token (BOS) from the reported metric.
    REQ-11's real Gemma run found this token's activation norm running
    roughly 8-9x every other token's at layer 20 -- a documented
    attention-sink outlier, not a reconstruction failure -- which alone
    was enough to drag a per-token fraction-variance-explained deeply
    negative even though every other token reconstructed well (0.63
    excluding it, versus catastrophically negative including it). This is
    the same convention published SAE evaluations use, not a Gemma-specific
    carve-out: excluding it is the correct read for either model, and
    ADR-0022 records the real before/after numbers for both rather than
    changing this silently.
    """
    import torch

    activations: list[torch.Tensor] = []

    def _capture(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
        activations.append(act.detach()[:, 1:, :])
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


_DEFAULT_FREQUENCY_CHUNK_SIZE = 64


def measure_activation_frequencies(
    loaded: LoadedPrismModel,
    token_batches: list["torch.Tensor"],
    *,
    chunk_size: int = _DEFAULT_FREQUENCY_CHUNK_SIZE,
) -> np.ndarray:
    """Return each SAE feature's activation rate over pre-tokenized text (REQ-2 / ADR-0011).

    A feature "activates" on a token when its encoded value is nonzero (the
    encoder's ReLU floor). Returns one rate per decoder atom, in the same
    row order as ``loaded.sae.W_dec``, so it lines up with ``decoder_norms()``
    and with the per-feature identifiability table without a remapping step.
    ``token_batches`` is pre-tokenized (and, for a large corpus, pre-truncated)
    by the caller -- this function only runs the forward passes and counts.

    Processes ``token_batches`` ``chunk_size`` documents at a time rather
    than concatenating and encoding the whole corpus in one pass, so peak
    memory is bounded by one chunk's activations and encoded output instead
    of the entire corpus's (issue #16: a dense encode() over the full
    ~111K-token corpus this project actually runs is a few gigabytes; a
    larger corpus without chunking risks an OOM). The result is
    mathematically identical to a single unchunked pass; only peak memory
    differs.
    """
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    nonzero_counts: torch.Tensor | None = None
    total_tokens = 0

    def _capture(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
        activations.append(act.detach())
        return act

    for start in range(0, len(token_batches), chunk_size):
        activations: list[torch.Tensor] = []
        with torch.no_grad():
            for tokens in token_batches[start : start + chunk_size]:
                loaded.model.run_with_hooks(tokens, fwd_hooks=[(loaded.hook_name, _capture)])

            acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
            fires = (loaded.sae.encode(acts) != 0).float()
        chunk_counts = fires.sum(dim=0).cpu()
        nonzero_counts = chunk_counts if nonzero_counts is None else nonzero_counts + chunk_counts
        total_tokens += acts.shape[0]

    if total_tokens == 0:
        raise ValueError("token_batches contains no tokens")
    return (nonzero_counts / total_tokens).numpy()


def decoder_norms(loaded: LoadedPrismModel) -> np.ndarray:
    """Return each SAE feature's raw, un-normalized decoder-vector norm.

    ADR-0010 records that this checkpoint's decoder atoms are not
    unit-normalized (``normalize_sae_decoder: false``), so this varies
    meaningfully across features and needs no separate measurement beyond
    reading ``W_dec`` directly.
    """
    import torch

    with torch.no_grad():
        norms = loaded.sae.W_dec.norm(dim=1)
    return norms.cpu().numpy()


def top_activating_snippets(
    loaded: LoadedPrismModel,
    feature_ids: list[int],
    token_batches: list["torch.Tensor"],
    *,
    k: int = 5,
    context_tokens: int = 8,
) -> dict[int, list[dict[str, Any]]]:
    """Return each requested feature's top-K activating context snippets (REQ-8, ADR-0018).

    Rather than a hand-written concept label, this grounds judge.py's
    naming-accuracy grading in the model's own real firing pattern: for
    each of ``feature_ids``, scan ``token_batches`` for the token positions
    where that feature's SAE-encoded value is highest, and return the
    surrounding ``context_tokens``-token window on each side as text --
    via the model's own tokenizer, so the snippet is exactly what the
    model saw, not a separate raw-text lookup that could drift out of
    alignment with the tokenization actually used to compute activations.

    Only ``feature_ids``' columns are ever materialized. Unlike
    ``measure_activation_frequencies()``, which needs every feature's rate,
    this only cares about a handful of REQ-2-sampled features, so there is
    no reason to encode the full dictionary width per token.

    Nearby positions within the same document can produce overlapping
    windows; this is accepted rather than deduplicated, since the judge
    only needs a sense of what the feature fires on, not maximally
    diverse coverage.

    Returns one entry per requested feature_id, each a list of
    ``{"activation": float, "snippet": str}`` sorted by activation
    descending, length at most ``k``. A feature that never fires anywhere
    in ``token_batches`` gets an empty list -- a real outcome (ADR-0013
    documents 77 of 16,384 features never firing over the full REQ-2
    corpus), not an error to guard against.
    """
    import heapq

    import torch

    if not feature_ids:
        raise ValueError("feature_ids must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")

    feature_index = torch.tensor(list(feature_ids), dtype=torch.long)
    heaps: dict[int, list[tuple[float, int, int]]] = {fid: [] for fid in feature_ids}

    for doc_idx, tokens in enumerate(token_batches):
        if tokens.shape[0] != 1:
            # After the reshape(-1, d_model) below, token_idx indexes the
            # flattened batch*seq axis, but the snippet-window slice further
            # down reads row 0 only (doc_tokens[0, start:end]) -- a batch
            # size > 1 would silently pull the snippet text from the wrong
            # row instead of raising. Every real caller (audit_build's
            # per-document corpus loader) already produces one row per
            # entry; this just makes that an enforced contract instead of
            # an unstated assumption.
            raise ValueError(
                f"token_batches[{doc_idx}] has batch size {tokens.shape[0]}; each entry must "
                "be a single document of shape (1, seq_len)"
            )
        activations: list[torch.Tensor] = []

        def _capture(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
            activations.append(act.detach())
            return act

        with torch.no_grad():
            loaded.model.run_with_hooks(tokens, fwd_hooks=[(loaded.hook_name, _capture)])
            acts = activations[0].reshape(-1, activations[0].shape[-1])
            encoded = loaded.sae.encode(acts)[:, feature_index]  # (seq_len, len(feature_ids))

        for column, fid in enumerate(feature_ids):
            values = encoded[:, column]
            for token_idx in range(values.shape[0]):
                value = values[token_idx].item()
                if not value > 0:
                    # Written as "not value > 0" rather than "value <= 0" on
                    # purpose: NaN comparisons are always False in both
                    # directions, so "value <= 0" silently lets a NaN
                    # activation through into the heap (where it would then
                    # corrupt heap ordering, since comparisons against NaN
                    # keep returning False there too). "not value > 0" is
                    # True for NaN, so it's skipped like any other
                    # non-positive value.
                    continue
                heap = heaps[fid]
                entry = (value, doc_idx, token_idx)
                if len(heap) < k:
                    heapq.heappush(heap, entry)
                elif value > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    results: dict[int, list[dict[str, Any]]] = {}
    for fid in feature_ids:
        ranked = sorted(heaps[fid], key=lambda entry: entry[0], reverse=True)
        snippets = []
        for value, doc_idx, token_idx in ranked:
            doc_tokens = token_batches[doc_idx]
            start = max(0, token_idx - context_tokens)
            end = min(doc_tokens.shape[1], token_idx + context_tokens + 1)
            snippets.append(
                {"activation": value, "snippet": loaded.model.to_string(doc_tokens[0, start:end])}
            )
        results[fid] = snippets
    return results


def save_reconstruction_result(
    config: dict[str, Any],
    result: dict[str, Any],
    output_path: Path = Path("data/results/req1_sae_validation.json"),
) -> Path:
    """Persist a validate_reconstruction() result with full run provenance.

    Per CLAUDE.md §2, a number that could end up in the report must be
    reconstructable from its logged config alone. validate_reconstruction()
    itself returns only n_tokens and fraction_variance_explained; this
    attaches the model/SAE checkpoint identity, git commit, and timestamp
    so the result isn't left as prose in a commit message.
    """
    import json
    import subprocess
    from datetime import datetime, timezone

    record = {
        "model_name": config["model"]["name"],
        "model_checkpoint_revision": config["model"]["checkpoint_revision"],
        "sae_checkpoint_repo": config["sae"]["checkpoint_repo"],
        "sae_checkpoint_revision": config["sae"]["checkpoint_revision"],
        "sae_checkpoint_sha256": config["sae"]["checkpoint_sha256"],
        "hook_name": config["sae"]["hook_name"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output_path
