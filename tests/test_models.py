"""Tests for prism.models — model + SAE loading (REQ-1)."""

from __future__ import annotations

from pathlib import Path
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
    top_activating_snippets,
    validate_reconstruction,
)

CONFIG_PATH = "configs/experiment.yaml"
GEMMA_CONFIG_PATH = "configs/experiment_gemma.yaml"


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


def test_gemma_config_has_no_remaining_model_or_sae_todos() -> None:
    # injection.strengths/features.n_total/sampling.seeds stay TODO pending
    # REQ-11's calibration pilot -- only the model/SAE loading fields, which
    # are already resolved, get checked here.
    with open(GEMMA_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"]["name"] != "TODO"
    assert config["model"]["checkpoint_revision"] != "TODO"
    for field in (
        "loader",
        "checkpoint_repo",
        "checkpoint_revision",
        "checkpoint_filename",
        "checkpoint_sha256",
        "sae_lens_release",
        "sae_lens_sae_id",
        "hook_name",
    ):
        assert config["sae"][field] != "TODO"


def test_load_model_and_sae_rejects_an_unrecognized_loader() -> None:
    config = {
        "model": {"name": "irrelevant", "checkpoint_revision": "irrelevant"},
        "sae": {"loader": "not_a_real_loader", "hook_name": "irrelevant"},
    }

    with pytest.raises(ValueError, match="unrecognized sae.loader"):
        load_model_and_sae(config)


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

    # 112, not 120: validate_reconstruction() excludes each prompt's BOS
    # token (ADR-0022) -- 8 prompts, 8 fewer tokens.
    assert result["n_tokens"] == 112
    # Measured 0.9782 on this checkpoint post-BOS-exclusion (was 0.9808
    # including it -- the BOS-outlier effect is real here too, just far
    # less severe than on Gemma); floor set below that with margin for
    # tokenizer/library version drift, not tuned to just barely pass.
    assert result["fraction_variance_explained"] >= 0.9

    output_path = save_reconstruction_result(config, result)
    assert output_path.exists()


@pytest.mark.integration
def test_load_model_and_sae_returns_a_working_gemma_pair() -> None:
    """Real, network-dependent load of REQ-11's Gemma Scope pair.

    Requires HF_TOKEN in the environment for an account that has accepted
    google/gemma-2-2b's gated license -- same posture as the Pythia
    integration test above (not a mock, per CLAUDE.md), just against a
    second model. Writes to its own output path so a run doesn't overwrite
    REQ-1's Pythia reconstruction result.
    """
    with open(GEMMA_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaded = load_model_and_sae(config)

    assert loaded.sae.cfg.d_in == loaded.model.cfg.d_model
    assert loaded.hook_name in loaded.model.hook_dict
    assert loaded.sae.W_dec.shape[0] == 16384

    result = validate_reconstruction(loaded, RECONSTRUCTION_VALIDATION_PROMPTS)

    # 121, not Pythia's 112 -- same 8 prompts (each losing its BOS token,
    # ADR-0022), but Gemma's tokenizer isn't Pythia's, so the same English
    # text doesn't tokenize to the same count. RECONSTRUCTION_VALIDATION_PROMPTS
    # is one canonical corpus shared across models; a fixed token count was
    # never part of that contract.
    assert result["n_tokens"] == 121
    assert 0.0 <= result["fraction_variance_explained"] <= 1.0

    output_path = save_reconstruction_result(
        config, result, output_path=Path("data/results/req1_gemma_sae_validation.json")
    )
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


# --- top_activating_snippets: concept grounding (REQ-8, ADR-0018) -------


def _fake_loaded_for_snippets(activations: list["torch.Tensor"]) -> LoadedPrismModel:
    """Same run_with_hooks/identity-encode fake as _fake_loaded_with_identity_encoder,
    plus a to_string() that renders a token-id tensor as space-joined integers, so
    snippet text is deterministic and inspectable without a real tokenizer.
    """
    activation_iterator = iter(activations)

    class _FakeModel:
        def run_with_hooks(self, tokens: object, fwd_hooks: list) -> None:
            del tokens
            ((_hook_name, hook_fn),) = fwd_hooks
            hook_fn(next(activation_iterator), None)

        def to_string(self, tensor: "torch.Tensor") -> str:
            return " ".join(str(i) for i in tensor.reshape(-1).tolist())

    class _FakeSAE:
        def encode(self, acts: "torch.Tensor") -> "torch.Tensor":
            return acts

    return LoadedPrismModel(model=_FakeModel(), sae=_FakeSAE(), hook_name="blocks.4.hook_resid_pre")


def test_top_activating_snippets_ranks_by_activation_descending() -> None:
    # One document, one feature (column 0), four tokens with distinct values.
    activations = [torch.tensor([[[0.1], [0.9], [0.3], [0.0]]])]
    token_batches = [torch.arange(4).reshape(1, 4)]

    result = top_activating_snippets(
        _fake_loaded_for_snippets(activations), [0], token_batches, k=2, context_tokens=1
    )

    assert [entry["snippet"] for entry in result[0]] == ["0 1 2", "1 2 3"]
    np.testing.assert_allclose([entry["activation"] for entry in result[0]], [0.9, 0.3], rtol=1e-6)


def test_top_activating_snippets_returns_empty_list_for_a_feature_that_never_fires() -> None:
    # SAE encode() output is a ReLU floor -- zero counts as "never fired," same
    # convention measure_activation_frequencies() uses for its nonzero check.
    activations = [torch.tensor([[[0.0], [0.0], [0.0]]])]
    token_batches = [torch.arange(3).reshape(1, 3)]

    result = top_activating_snippets(_fake_loaded_for_snippets(activations), [0], token_batches, k=5)

    assert result[0] == []


def test_top_activating_snippets_spans_multiple_documents() -> None:
    # Two one-token documents; the stronger activation lives in the second
    # document, so its snippet must be drawn from that document's own tokens,
    # not the first document's, even though it's ranked first in the result.
    activations = [torch.tensor([[[0.5]]]), torch.tensor([[[0.8]]])]
    token_batches = [torch.tensor([[100]]), torch.tensor([[200]])]

    result = top_activating_snippets(
        _fake_loaded_for_snippets(activations), [0], token_batches, k=5, context_tokens=2
    )

    assert [entry["snippet"] for entry in result[0]] == ["200", "100"]
    np.testing.assert_allclose([entry["activation"] for entry in result[0]], [0.8, 0.5], rtol=1e-6)


def test_top_activating_snippets_clips_the_window_at_document_boundaries() -> None:
    activations = [torch.tensor([[[0.7], [0.2]]])]
    token_batches = [torch.tensor([[10, 20]])]

    result = top_activating_snippets(
        _fake_loaded_for_snippets(activations), [0], token_batches, k=5, context_tokens=10
    )

    # context_tokens=10 would run off both ends of a 2-token document; the
    # window must clip to what the document actually has, not raise or pad.
    assert result[0][0]["snippet"] == "10 20"


def test_top_activating_snippets_rejects_empty_feature_ids() -> None:
    with pytest.raises(ValueError, match="feature_ids must be non-empty"):
        top_activating_snippets(_fake_loaded_for_snippets([]), [], [])


def test_top_activating_snippets_rejects_non_positive_k() -> None:
    activations = [torch.tensor([[[0.5]]])]
    token_batches = [torch.tensor([[1]])]

    with pytest.raises(ValueError, match="k must be positive"):
        top_activating_snippets(_fake_loaded_for_snippets(activations), [0], token_batches, k=0)


def test_top_activating_snippets_skips_nan_activations() -> None:
    # NaN fails both "<= 0" and "> 0" comparisons, so a naive non-positive
    # check lets it slip into the heap and corrupt ordering; only the real
    # 0.5 value should survive here.
    nan = float("nan")
    activations = [torch.tensor([[[nan], [0.5], [nan]]])]
    token_batches = [torch.arange(3).reshape(1, 3)]

    result = top_activating_snippets(_fake_loaded_for_snippets(activations), [0], token_batches, k=5)

    assert len(result[0]) == 1
    assert result[0][0]["activation"] == 0.5


def test_top_activating_snippets_rejects_a_multi_row_token_batch() -> None:
    # After the reshape(-1, d_model), token_idx indexes the flattened
    # batch*seq axis, but the snippet window later reads row 0 only -- a
    # batch size > 1 would silently pull text from the wrong row rather
    # than raise. The check fires before any hook call, so an empty
    # activations list is fine here.
    token_batches = [torch.tensor([[1], [2]])]  # shape (2, 1): batch size 2

    with pytest.raises(ValueError, match="batch size"):
        top_activating_snippets(_fake_loaded_for_snippets([]), [0], token_batches, k=5)


@pytest.mark.integration
def test_top_activating_snippets_against_the_real_pair() -> None:
    """Real model/SAE, small prompt set: shapes and value ranges only, the
    same integration-test posture as REQ-1/REQ-2's own real-pair checks.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaded = load_model_and_sae(config)
    token_batches = [loaded.model.to_tokens(prompt) for prompt in RECONSTRUCTION_VALIDATION_PROMPTS]

    result = top_activating_snippets(loaded, [0, 1], token_batches, k=3, context_tokens=4)

    assert set(result.keys()) == {0, 1}
    for snippets in result.values():
        assert len(snippets) <= 3
        for entry in snippets:
            assert entry["activation"] > 0
            assert isinstance(entry["snippet"], str) and entry["snippet"]


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
