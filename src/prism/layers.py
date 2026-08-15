"""Injection-layer selection (REQ-10, stretch) and its ADR-0009 fallback.

``get_compression_boundary_layer()`` -- the geometry-grounded lookup against
a precomputed UCARE intrinsic-dimension trajectory -- is REQ-10's own scope
and is not implemented here yet; it needs that trajectory as an external
input, the same way ``data/audit/features.csv`` is treated as read-only
input rather than something this project regenerates.

``get_fallback_layer()`` is the one piece of ADR-0009 that's already fully
specified independent of that trajectory: a fixed fraction of the model's
depth, used until REQ-10 resolves the primary layer for real. REQ-5's
calibration pilot calls this now, since it needs some layer to inject into
and REQ-10 hasn't run yet -- ADR-0009 names this an explicit fallback, not
an approximation of the geometry-grounded choice, so callers should record
which one they used rather than letting the two blur together.
"""

from __future__ import annotations

DEFAULT_FALLBACK_FRACTION = 2 / 3


def get_fallback_layer(n_layers: int, fraction: float = DEFAULT_FALLBACK_FRACTION) -> int:
    """Return the ADR-0009 fractional-depth fallback layer for a model this deep.

    Rounds ``n_layers * fraction`` to the nearest block index (Python's
    round-half-to-even), then clamps to ``n_layers - 1`` so a ``fraction``
    of exactly ``1.0`` still names a real block rather than one past the
    model's last layer -- valid ``blocks.{i}.hook_resid_pre`` indices run
    ``0`` through ``n_layers - 1``.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    layer = round(n_layers * fraction)
    return min(layer, n_layers - 1)
