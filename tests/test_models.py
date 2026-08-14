"""Tests for prism.models — model + SAE loading (REQ-1)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from prism.models import (
    RECONSTRUCTION_VALIDATION_PROMPTS,
    LoadedPrismModel,
    _validate_pairing,
    _verify_sha256,
    decoder_norms,
    load_model_and_sae,
    measure_activation_frequencies,
    save_reconstruction_result,
    validate_reconstruction,
)

CONFIG_PATH = "configs/experiment.yaml"


def _fake_model(hook_names: list[str], d_model: int) -> SimpleNamespace:
    return SimpleNamespace(hook_dict=dict.fromkeys(hook_names), cfg=SimpleNamespace(d_model=d_model))


def _fake_sae(trained_hook_name: str, d_in: int) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(d_in=d_in, metadata=SimpleNamespace(hook_name=trained_hook_name))
    )


def test_validate_pairing_accepts_matching_model_and_sae() -> None:
    model = _fake_model(["blocks.4.hook_resid_pre"], d_model=512)
    sae = _fake_sae("blocks.4.hook_resid_pre", d_in=512)

    _validate_pairing(model, sae, "blocks.4.hook_resid_pre")


def test_validate_pairing_rejects_hook_name_absent_from_model() -> None:
    model = _fake_model(["blocks.3.hook_resid_pre"], d_model=512)
    sae = _fake_sae("blocks.4.hook_resid_pre", d_in=512)

    with pytest.raises(ValueError, match="does not exist on"):
        _validate_pairing(model, sae, "blocks.4.hook_resid_pre")


def test_validate_pairing_rejects_hook_name_the_sae_was_not_trained_on() -> None:
    # Hook exists on the model, but isn't the one this SAE's dictionary was scored against.
    model = _fake_model(["blocks.3.hook_resid_pre", "blocks.4.hook_resid_pre"], d_model=512)
    sae = _fake_sae("blocks.4.hook_resid_pre", d_in=512)

    with pytest.raises(ValueError, match="does not match the hook"):
        _validate_pairing(model, sae, "blocks.3.hook_resid_pre")


def test_validate_pairing_rejects_dimension_mismatch() -> None:
    # Same hook name, but a decoder trained against a different model's width --
    # exactly the "wrong Pythia checkpoint" confound ADR-0002 warns about.
    model = _fake_model(["blocks.4.hook_resid_pre"], d_model=512)
    sae = _fake_sae("blocks.4.hook_resid_pre", d_in=768)

    with pytest.raises(ValueError, match="d_in"):
        _validate_pairing(model, sae, "blocks.4.hook_resid_pre")


_DETERMINISTIC_CONTENT_SHA256 = "637f557ec73a25a2aec3b6dedf45705a0a0c2bffd90f102911df85501a1a547f"


def test_verify_sha256_accepts_matching_digest(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"deterministic content")

    _verify_sha256(path, _DETERMINISTIC_CONTENT_SHA256)  # must not raise


def test_verify_sha256_rejects_mismatched_digest(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"tampered content")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_sha256(path, _DETERMINISTIC_CONTENT_SHA256)


def test_config_has_no_remaining_model_or_sae_todos() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"]["name"] != "TODO"
    for field in (
        "checkpoint_repo",
        "checkpoint_revision",
        "checkpoint_subfolder",
        "checkpoint_sha256",
        "checkpoint_cfg_sha256",
    ):
        assert config["sae"][field] != "TODO"


@pytest.mark.integration
def test_load_model_and_sae_returns_a_working_pair() -> None:
    """Real, network-dependent load against the resolved checkpoint (ADR-0010).

    Not a mock: this is the actual model and the actual audited SAE, per
    CLAUDE.md's rule against faking pipeline stages. Slow and network-bound,
    which is why it's marked separately from the fast unit tests above.

    Also exercises validate_reconstruction() against the fixed corpus REQ-1's
    definition of done requires reconstruction quality for, and persists the
    result so it's a traceable artifact (git commit, timestamp, checkpoint
    identity) rather than only prose in a commit message.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaded = load_model_and_sae(config)

    assert loaded.sae.cfg.d_in == loaded.model.cfg.d_model
    assert loaded.hook_name in loaded.model.hook_dict

    result = validate_reconstruction(loaded, RECONSTRUCTION_VALIDATION_PROMPTS)

    assert result["n_tokens"] == 120
    # Measured 0.9808 on this checkpoint; floor set below that with margin for
    # tokenizer/library version drift, not tuned to just barely pass.
    assert result["fraction_variance_explained"] >= 0.9

    output_path = save_reconstruction_result(config, result)
    assert output_path.exists()


def _fake_loaded_for_frequency(activations: list, encoded_output: "torch.Tensor") -> LoadedPrismModel:
    """A LoadedPrismModel whose model/sae are fakes, not real TransformerLens/SAELens
    objects, matching this file's existing _fake_model/_fake_sae convention above.

    ``encode()`` is called exactly once, on every batch's activations
    concatenated together, so ``encoded_output`` must have one row per
    token across all of ``activations`` -- not one value per call.
    """
    activation_iterator = iter(activations)

    class _FakeModel:
        def run_with_hooks(self, tokens: object, fwd_hooks: list) -> None:
            del tokens
            ((_hook_name, hook_fn),) = fwd_hooks
            hook_fn(next(activation_iterator), None)

    class _FakeSAE:
        def encode(self, acts: "torch.Tensor") -> "torch.Tensor":
            assert acts.shape[0] == encoded_output.shape[0]
            return encoded_output

    return LoadedPrismModel(model=_FakeModel(), sae=_FakeSAE(), hook_name="blocks.4.hook_resid_pre")


def _fake_loaded_with_identity_encoder(activations: list) -> LoadedPrismModel:
    """A LoadedPrismModel whose fake SAE returns whatever it's handed, unchanged.

    Unlike _fake_loaded_for_frequency above (one fixed encode() output for
    a single call over the whole corpus), this supports encode() being
    called an arbitrary number of times with arbitrary chunk sizes, which
    chunked measure_activation_frequencies() calls need to exercise.
    """
    activation_iterator = iter(activations)

    class _FakeModel:
        def run_with_hooks(self, tokens: object, fwd_hooks: list) -> None:
            del tokens
            ((_hook_name, hook_fn),) = fwd_hooks
            hook_fn(next(activation_iterator), None)

    class _FakeSAE:
        def encode(self, acts: "torch.Tensor") -> "torch.Tensor":
            return acts

    return LoadedPrismModel(model=_FakeModel(), sae=_FakeSAE(), hook_name="blocks.4.hook_resid_pre")


def test_measure_activation_frequencies_chunking_matches_unchunked() -> None:
    # Four one-token documents; the fake SAE passes activations through
    # unchanged, so each document's own values determine which "features"
    # (columns) count as active. chunk_size=1 forces four separate encode()
    # calls; chunk_size=1000 forces one, covering the whole corpus.
    activations = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
    ]
    token_batches = [torch.zeros(1, 1, dtype=torch.long) for _ in activations]

    chunked = measure_activation_frequencies(
        _fake_loaded_with_identity_encoder(activations), token_batches, chunk_size=1
    )
    unchunked = measure_activation_frequencies(
        _fake_loaded_with_identity_encoder(activations), token_batches, chunk_size=1000
    )

    np.testing.assert_allclose(chunked, unchunked)
    np.testing.assert_allclose(chunked, [0.5, 0.5])


def test_measure_activation_frequencies_rejects_non_positive_chunk_size() -> None:
    loaded = _fake_loaded_with_identity_encoder([torch.tensor([[1.0]])])

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        measure_activation_frequencies(loaded, [torch.zeros(1, 1, dtype=torch.long)], chunk_size=0)


def test_measure_activation_frequencies_counts_nonzero_encoded_activations() -> None:
    # Two one-token batches, concatenated to two rows before the single
    # encode() call; the fake SAE "encodes" that to a fixed 3-feature matrix
    # so the expected per-feature rate can be computed by hand.
    loaded = _fake_loaded_for_frequency(
        activations=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)],
        encoded_output=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
    )
    token_batches = [torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long)]

    rates = measure_activation_frequencies(loaded, token_batches)

    np.testing.assert_allclose(rates, [0.5, 0.0, 0.5])


def test_measure_activation_frequencies_handles_multi_token_batches() -> None:
    loaded = _fake_loaded_for_frequency(
        activations=[torch.zeros(1, 4, 2)],
        encoded_output=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]),
    )

    rates = measure_activation_frequencies(loaded, [torch.zeros(1, 4, dtype=torch.long)])

    np.testing.assert_allclose(rates, [0.5, 0.25])


def test_decoder_norms_returns_row_wise_norm_of_w_dec() -> None:
    w_dec = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])  # norms: 5.0, 1.0, 0.0

    loaded = LoadedPrismModel(
        model=SimpleNamespace(),
        sae=SimpleNamespace(W_dec=w_dec),
        hook_name="blocks.4.hook_resid_pre",
    )

    norms = decoder_norms(loaded)

    np.testing.assert_allclose(norms, [5.0, 1.0, 0.0])


@pytest.mark.integration
def test_measure_activation_frequencies_and_decoder_norms_against_the_real_pair() -> None:
    """Real model/SAE, small prompt set: shapes and value ranges only.

    The full-corpus measurement (REQ-2's actual activation-frequency run)
    is produced by prism.audit_build against a much larger pinned corpus;
    this just confirms the mechanism works end to end against the real
    checkpoint pair, the same way REQ-1's integration test does for
    validate_reconstruction().
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaded = load_model_and_sae(config)
    n_features = loaded.sae.W_dec.shape[0]
    token_batches = [loaded.model.to_tokens(prompt) for prompt in RECONSTRUCTION_VALIDATION_PROMPTS]

    rates = measure_activation_frequencies(loaded, token_batches)
    norms = decoder_norms(loaded)

    assert rates.shape == (n_features,)
    assert norms.shape == (n_features,)
    assert (rates >= 0).all() and (rates <= 1).all()
    assert (norms >= 0).all()
