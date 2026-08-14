"""Tests for prism.models — model + SAE loading (REQ-1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from prism.models import _validate_pairing, _verify_sha256, load_model_and_sae

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
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaded = load_model_and_sae(config)

    assert loaded.sae.cfg.d_in == loaded.model.cfg.d_model
    assert loaded.hook_name in loaded.model.hook_dict
