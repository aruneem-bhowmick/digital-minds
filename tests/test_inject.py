"""Tests for prism.inject -- the injection hook (REQ-3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml

from prism.inject import inject_concept, no_injection
from prism.models import load_model_and_sae

CONFIG_PATH = "configs/experiment.yaml"
LAYER = 4
HOOK_NAME = "blocks.4.hook_resid_pre"


def _fake_model(hook_names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(hook_dict=dict.fromkeys(hook_names))


# --- inject_concept: validation ---------------------------------------------


def test_inject_concept_rejects_layer_without_a_matching_hook_point() -> None:
    model = _fake_model(["blocks.3.hook_resid_pre"])

    with pytest.raises(ValueError, match="hook_resid_pre"):
        inject_concept(model, torch.tensor([1.0, 0.0]), layer=LAYER, strength=2.0, token_start_pos=0)


def test_inject_concept_rejects_negative_token_start_pos() -> None:
    model = _fake_model([HOOK_NAME])

    with pytest.raises(ValueError, match="token_start_pos"):
        inject_concept(model, torch.tensor([1.0, 0.0]), layer=LAYER, strength=2.0, token_start_pos=-1)


# --- inject_concept / no_injection: passthrough -----------------------------


def test_inject_concept_returns_no_hooks_when_atom_is_none() -> None:
    model = _fake_model([HOOK_NAME])

    hooks = inject_concept(model, None, layer=LAYER, strength=2.0, token_start_pos=3)

    assert hooks == []


def test_no_injection_returns_no_hooks() -> None:
    model = _fake_model([HOOK_NAME])

    assert no_injection(model, layer=LAYER, token_start_pos=3) == []


def test_no_injection_does_not_require_a_valid_layer() -> None:
    # Interface symmetry with inject_concept means callers can pass whatever
    # layer/token_start_pos the injected-trial branch would have used --
    # no_injection() never validates them, since there is no hook to attach.
    model = _fake_model([])

    assert no_injection(model, layer=99, token_start_pos=-1) == []


# --- inject_concept: normalization and scaling ------------------------------


def test_inject_concept_normalizes_before_scaling() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([3.0, 4.0])  # norm 5

    ((hook_name, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=2.0, token_start_pos=0)
    out = hook_fn(torch.zeros(1, 1, 2), None)

    assert hook_name == HOOK_NAME
    # normalized atom = [0.6, 0.8]; strength 2.0 -> [1.2, 1.6]
    torch.testing.assert_close(out, torch.tensor([[[1.2, 1.6]]]))


def test_inject_concept_zero_vector_atom_is_a_no_op() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.zeros(2)

    ((_, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=9.0, token_start_pos=0)
    act = torch.tensor([[[1.0, -2.0]]])
    out = hook_fn(act, None)

    torch.testing.assert_close(out, act)


def test_inject_concept_zero_strength_is_a_no_op() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([3.0, 4.0])

    ((_, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=0.0, token_start_pos=0)
    act = torch.tensor([[[1.0, -2.0]]])
    out = hook_fn(act, None)

    torch.testing.assert_close(out, act)


def test_inject_concept_negative_strength() -> None:
    # N/A: REQ-5's calibration pilot only ever hands inject_concept() a
    # positive, calibrated strength (SPRINT-PLAN.md §3.3); nothing upstream
    # of this function can produce a negative one to exercise here.
    pass


# --- inject_concept: persistence across a chunked, cached generation call ---


def test_inject_concept_prefill_chunk_only_injects_at_the_start_position() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([1.0, 0.0])

    # A 3-token prompt fed in one prefill chunk; token_start_pos=2 is the
    # last of those three positions ("immediately before the model's
    # response", per ADR-0003).
    ((_, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=1.0, token_start_pos=2)
    out = hook_fn(torch.zeros(1, 3, 2), None)

    torch.testing.assert_close(out[:, :2, :], torch.zeros(1, 2, 2))
    torch.testing.assert_close(out[:, 2, :], torch.tensor([[1.0, 0.0]]))


def test_inject_concept_persists_across_every_decode_step_after_start() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([1.0, 0.0])

    ((_, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=1.0, token_start_pos=2)
    hook_fn(torch.zeros(1, 3, 2), None)  # prefill: positions 0, 1, 2

    for _ in range(4):  # four more decode steps: positions 3, 4, 5, 6
        out = hook_fn(torch.zeros(1, 1, 2), None)
        torch.testing.assert_close(out, torch.tensor([[[1.0, 0.0]]]))


def test_inject_concept_does_not_inject_before_the_start_position() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([1.0, 0.0])

    # token_start_pos=5 is past every position in this 3-token chunk.
    ((_, hook_fn),) = inject_concept(model, atom, layer=LAYER, strength=1.0, token_start_pos=5)
    out = hook_fn(torch.zeros(1, 3, 2), None)

    torch.testing.assert_close(out, torch.zeros(1, 3, 2))


# --- inject_concept: state isolation across separate calls ------------------


def test_inject_concept_position_state_does_not_leak_between_calls() -> None:
    model = _fake_model([HOOK_NAME])
    atom = torch.tensor([1.0, 0.0])

    ((_, first_hook),) = inject_concept(model, atom, layer=LAYER, strength=1.0, token_start_pos=0)
    first_hook(torch.zeros(1, 5, 2), None)  # advance the first call's internal position to 5

    # A fresh inject_concept() call must start counting from zero again,
    # not continue from the first call's ending position.
    ((_, second_hook),) = inject_concept(model, atom, layer=LAYER, strength=1.0, token_start_pos=2)
    out = second_hook(torch.zeros(1, 3, 2), None)

    torch.testing.assert_close(out[:, :2, :], torch.zeros(1, 2, 2))
    torch.testing.assert_close(out[:, 2, :], torch.tensor([[1.0, 0.0]]))


# --- integration: real model, real SAE, real generate() ---------------------

_PROMPT = "The weather today is"
_MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def loaded_pair():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return load_model_and_sae(config)


def _generate(model, tokens: "torch.Tensor") -> "torch.Tensor":
    return model.generate(tokens, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False, verbose=False)


@pytest.mark.integration
def test_inject_concept_zero_vector_reproduces_baseline_generation_exactly(loaded_pair) -> None:
    model = loaded_pair.model
    tokens = model.to_tokens(_PROMPT)
    token_start_pos = tokens.shape[1] - 1

    baseline = _generate(model, tokens)

    zero_atom = torch.zeros_like(loaded_pair.sae.W_dec[0])
    hooks = inject_concept(model, zero_atom, layer=LAYER, strength=5.0, token_start_pos=token_start_pos)
    with model.hooks(fwd_hooks=hooks):
        zero_injected = _generate(model, tokens)

    assert torch.equal(zero_injected, baseline)


@pytest.mark.integration
def test_inject_concept_zero_strength_reproduces_baseline_generation_exactly(loaded_pair) -> None:
    model = loaded_pair.model
    tokens = model.to_tokens(_PROMPT)
    token_start_pos = tokens.shape[1] - 1

    baseline = _generate(model, tokens)

    real_atom = loaded_pair.sae.W_dec[0]
    hooks = inject_concept(model, real_atom, layer=LAYER, strength=0.0, token_start_pos=token_start_pos)
    with model.hooks(fwd_hooks=hooks):
        zero_strength = _generate(model, tokens)

    assert torch.equal(zero_strength, baseline)


@pytest.mark.integration
def test_no_injection_reproduces_baseline_generation_exactly(loaded_pair) -> None:
    model = loaded_pair.model
    tokens = model.to_tokens(_PROMPT)
    token_start_pos = tokens.shape[1] - 1

    baseline = _generate(model, tokens)

    hooks = no_injection(model, layer=LAYER, token_start_pos=token_start_pos)
    with model.hooks(fwd_hooks=hooks):
        passthrough = _generate(model, tokens)

    assert torch.equal(passthrough, baseline)


@pytest.mark.integration
def test_inject_concept_leaves_no_hooks_attached_after_generation(loaded_pair) -> None:
    model = loaded_pair.model
    tokens = model.to_tokens(_PROMPT)
    token_start_pos = tokens.shape[1] - 1

    assert model.hook_dict[HOOK_NAME].fwd_hooks == []

    hooks = inject_concept(
        model, loaded_pair.sae.W_dec[0], layer=LAYER, strength=3.0, token_start_pos=token_start_pos
    )
    with model.hooks(fwd_hooks=hooks):
        assert len(model.hook_dict[HOOK_NAME].fwd_hooks) == 1
        _generate(model, tokens)

    assert model.hook_dict[HOOK_NAME].fwd_hooks == []


@pytest.mark.integration
def test_inject_concept_cleans_up_hooks_even_if_the_call_raises(loaded_pair) -> None:
    model = loaded_pair.model
    token_start_pos = 0

    hooks = inject_concept(
        model, loaded_pair.sae.W_dec[0], layer=LAYER, strength=3.0, token_start_pos=token_start_pos
    )

    with pytest.raises(RuntimeError, match="boom"):
        with model.hooks(fwd_hooks=hooks):
            assert len(model.hook_dict[HOOK_NAME].fwd_hooks) == 1
            raise RuntimeError("boom")

    assert model.hook_dict[HOOK_NAME].fwd_hooks == []
