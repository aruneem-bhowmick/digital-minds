"""Injection hook and strength calibration (REQ-3, REQ-5).

REQ-3 / ADR-0001 / ADR-0003: ``inject_concept`` builds a TransformerLens
forward hook that adds a scaled, normalized SAE decoder atom into a chosen
layer's residual stream, starting at the token position immediately before
the model's response and persisting through every token generated after
it -- not a single-token nudge. It returns the ``(hook_name, hook_fn)`` pair
TransformerLens's ``run_with_hooks`` / ``model.hooks()`` pattern expects;
this module never drives generation itself, so callers control their own
``model.generate()`` arguments (temperature, seed, max_new_tokens).

Strength calibration (REQ-5) is a separate concern layered on top of this
mechanism, not implemented here. This module works correctly with any
placeholder strength a caller supplies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from transformer_lens import HookedTransformer


def inject_concept(
    model: "HookedTransformer",
    decoder_atom: "torch.Tensor | None",
    layer: int,
    strength: float,
    token_start_pos: int,
) -> "list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]":
    """Build the forward hook for a persistent concept injection.

    ``decoder_atom=None`` is the no-injection passthrough (REQ-7's baseline
    trials): returns an empty hook list rather than a hook that happens to
    add zero, so callers always write
    ``with model.hooks(fwd_hooks=inject_concept(...)): model.generate(...)``
    with no branching on whether a trial is injected.

    Raises if ``layer`` has no matching residual-stream hook point on
    ``model`` -- a wrong-layer config error needs to surface here, before a
    hook is ever attached, not several calls downstream during generation.
    """
    if decoder_atom is None:
        return []

    if token_start_pos < 0:
        raise ValueError(f"token_start_pos must be non-negative, got {token_start_pos}")

    hook_name = _resid_pre_hook_name(layer)
    if hook_name not in model.hook_dict:
        raise ValueError(
            f"layer {layer} has no {hook_name!r} hook point on this model; "
            "check the configured injection layer against the model's depth"
        )

    injected_vector = _scaled_atom(decoder_atom, strength)
    return [(hook_name, _make_injection_hook(injected_vector, token_start_pos))]


def no_injection(
    model: "HookedTransformer", layer: int, token_start_pos: int
) -> "list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]":
    """Explicit no-injection passthrough, sharing ``inject_concept``'s argument
    shape (REQ-7) so a baseline-trial call site reads the same as an injected
    one, differing only in which function name it calls.
    """
    del layer, token_start_pos  # kept for interface symmetry with inject_concept, unused here
    return []


def _resid_pre_hook_name(layer: int) -> str:
    """The residual-stream hook point this project injects into.

    ``hook_resid_pre`` mirrors the SAE's own hook-point convention
    (``configs/experiment.yaml``'s ``sae.hook_name``) and standard
    activation-steering practice: the vector is added to the residual
    stream immediately entering the chosen layer, the same site a decoder
    atom's own dictionary was fit against.
    """
    return f"blocks.{layer}.hook_resid_pre"


def _scaled_atom(decoder_atom: "torch.Tensor", strength: float) -> "torch.Tensor":
    """Normalize before scaling, per ADR-0003.

    ADR-0010 recorded that this checkpoint's decoder atoms are not
    unit-normalized in the saved weights, so raw atom norms vary across the
    dictionary and are not comparable until each one is put on the same
    scale first. A genuinely zero-norm atom skips the division (which would
    otherwise be 0/0) and is returned as-is: scaling an all-zero vector by
    any strength is still all zero.
    """
    norm = decoder_atom.norm()
    if norm == 0:
        return decoder_atom
    return strength * (decoder_atom / norm)


def _make_injection_hook(
    injected_vector: "torch.Tensor", token_start_pos: int
) -> "Callable[[torch.Tensor, Any], torch.Tensor]":
    """Return a hook function that adds ``injected_vector`` to every position
    at or beyond ``token_start_pos``, and keeps doing so across every later
    call within the same generation.

    TransformerLens's ``generate()`` drives one forward call for the whole
    prompt, then one call per newly generated token once its KV cache is
    primed, and hands the hook only that call's chunk of the residual
    stream -- never the token's absolute position in the full sequence.
    This closure tracks how many positions it has already seen, starting
    fresh at zero every time ``inject_concept()`` is called (so nothing
    survives between trials), and advances that count by each chunk's
    length. That is exactly right for both the first, full-prompt call and
    every later, single-new-token call a cached ``generate()`` makes; this
    project's generation always goes through the cache, so a raw,
    uncached forward pass that reprocesses a growing sequence from scratch
    each call is not a case this hook needs to handle.
    """
    seen = 0

    def _hook(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
        nonlocal seen
        del hook
        chunk_len = act.shape[1]
        start, end = seen, seen + chunk_len
        seen = end

        if end <= token_start_pos:
            return act  # whole chunk still precedes the injection start

        vector = injected_vector.to(device=act.device, dtype=act.dtype)
        if start >= token_start_pos:
            return act + vector  # whole chunk is at or past the start position

        first_injected_local_pos = token_start_pos - start
        act = act.clone()
        act[:, first_injected_local_pos:, :] += vector
        return act

    return _hook
