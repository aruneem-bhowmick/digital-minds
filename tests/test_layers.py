"""Tests for prism.layers -- the ADR-0009 fallback layer (REQ-5)."""

from __future__ import annotations

import pytest

from prism.layers import get_fallback_layer, resolve_injection_layer

# --- get_fallback_layer: normal cases ----------------------------------------


def test_get_fallback_layer_default_fraction_on_pythia_70m_depth() -> None:
    # Pythia-70m-deduped has 6 transformer blocks; 6 * 2/3 = 4.0 exactly,
    # which happens to match the layer this project's SAE was already
    # trained against (blocks.4.hook_resid_pre).
    assert get_fallback_layer(6) == 4


def test_get_fallback_layer_rounds_to_nearest_block() -> None:
    # 10 * 2/3 = 6.666..., rounds up to 7.
    assert get_fallback_layer(10) == 7


def test_get_fallback_layer_accepts_a_custom_fraction() -> None:
    assert get_fallback_layer(12, fraction=0.5) == 6


def test_get_fallback_layer_single_layer_model() -> None:
    assert get_fallback_layer(1) == 0


# --- get_fallback_layer: clamping to a valid block index ---------------------


def test_get_fallback_layer_clamps_a_fraction_of_one_to_the_last_block() -> None:
    # 6 * 1.0 = 6, which is one past the last valid block index (5).
    assert get_fallback_layer(6, fraction=1.0) == 5


# --- get_fallback_layer: validation -------------------------------------------


def test_get_fallback_layer_rejects_zero_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        get_fallback_layer(0)


def test_get_fallback_layer_rejects_negative_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        get_fallback_layer(-3)


def test_get_fallback_layer_rejects_zero_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        get_fallback_layer(6, fraction=0)


def test_get_fallback_layer_rejects_fraction_above_one() -> None:
    with pytest.raises(ValueError, match="fraction"):
        get_fallback_layer(6, fraction=1.5)


def test_get_fallback_layer_rejects_negative_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        get_fallback_layer(6, fraction=-0.5)


# --- resolve_injection_layer ---------------------------------------------


def test_resolve_injection_layer_falls_back_when_layer_is_the_todo_string() -> None:
    config = {"injection": {"layer": "TODO"}}

    layer, layer_source = resolve_injection_layer(config, n_layers=6)

    assert layer == get_fallback_layer(6)
    assert layer_source == "adr-0009-fallback"


def test_resolve_injection_layer_falls_back_when_the_injection_key_is_missing() -> None:
    # A test fixture config with no injection block at all -- the same
    # "not resolved yet" state as the TODO string, not a separate case a
    # caller needs to handle differently.
    config: dict = {}

    layer, layer_source = resolve_injection_layer(config, n_layers=6)

    assert layer == get_fallback_layer(6)
    assert layer_source == "adr-0009-fallback"


def test_resolve_injection_layer_honors_an_explicit_layer() -> None:
    # REQ-11's Gemma Scope config: injection.layer: 20, fixed by which SAE
    # checkpoint exists, not the ADR-0009 fractional-depth fallback.
    config = {"injection": {"layer": 20}}

    layer, layer_source = resolve_injection_layer(config, n_layers=26)

    assert layer == 20
    assert layer_source == "sae-checkpoint-layer"


def test_resolve_injection_layer_falls_back_when_the_injection_value_is_none() -> None:
    # A YAML `injection:` stanza with no nested keys parses to
    # {"injection": None}, not a missing key -- config.get("injection", {})
    # returns None (not the {} default) in that case, so this needs its own
    # test distinct from the missing-key case above.
    config = {"injection": None}

    layer, layer_source = resolve_injection_layer(config, n_layers=6)

    assert layer == get_fallback_layer(6)
    assert layer_source == "adr-0009-fallback"


def test_resolve_injection_layer_honors_layer_zero() -> None:
    # layer=0 is a legitimate, valid explicit layer -- must not be treated
    # like the falsy-but-unresolved None/"TODO" cases above.
    config = {"injection": {"layer": 0}}

    layer, layer_source = resolve_injection_layer(config, n_layers=6)

    assert layer == 0
    assert layer_source == "sae-checkpoint-layer"


def test_resolve_injection_layer_accepts_a_quoted_numeric_string() -> None:
    # A config typo like `layer: "20"` (quoted) must resolve to layer 20,
    # not silently fall back -- only the literal "TODO" sentinel means
    # "unresolved," not "any string value."
    config = {"injection": {"layer": "20"}}

    layer, layer_source = resolve_injection_layer(config, n_layers=26)

    assert layer == 20
    assert layer_source == "sae-checkpoint-layer"


def test_resolve_injection_layer_rejects_a_layer_at_or_above_n_layers() -> None:
    config = {"injection": {"layer": 26}}  # valid range for a 26-layer model is 0..25

    with pytest.raises(ValueError, match="out of range"):
        resolve_injection_layer(config, n_layers=26)


def test_resolve_injection_layer_rejects_a_negative_layer() -> None:
    config = {"injection": {"layer": -1}}

    with pytest.raises(ValueError, match="out of range"):
        resolve_injection_layer(config, n_layers=26)
