"""Detection, naming, and control prompt templates (REQ-4).

SPRINT-PLAN.md §3.4 calls for three pieces: a detection question, a naming
follow-up, and an unrelated-question control. ADR-0015 records where the
control-question set lives and how each function maps onto Lindsey's
introspection criteria, using the framing the REQ-4 build sequence itself
names: detection, naming/accuracy, internality, and coherence. Coherence has
no dedicated template -- REQ-8's judge scores it from whatever text a prompt
elicits, not from a specific question.

None of the wording below reproduces Lindsey (2025)'s own prompt text; each
function serves the same experimental role with independently written
phrasing.
"""

from __future__ import annotations


def detection_prompt() -> str:
    """Return the primary detection question (SPRINT-PLAN.md §3.4, "Detection trial").

    Targets Lindsey's detection criterion: whether the model reports
    noticing anything unusual at all. The wording never mentions injection,
    steering, or an external cause, so the identical prompt is valid for
    both REQ-6's injected trials and REQ-7's no-injection baseline -- a
    baseline built from a prompt that already presupposes an injection
    would not measure a genuine false-positive rate.

    Asking for a plain yes-or-no verdict before any description is
    deliberate (ADR-0015): it keeps detection separable from naming, which
    is what makes "detection before verbalization" -- Morris & Plunkett's
    causal-bypassing criterion, SPRINT-PLAN.md line 33 -- checkable from the
    transcript, rather than assumed.
    """
    return (
        "Before you respond, pause and check your own current processing. "
        "Is there anything about it that feels unusual right now, "
        "something that doesn't seem to follow from this conversation on "
        "its own? Answer with a single word first, yes or no, and hold off "
        "on describing anything further until asked."
    )


def naming_subtask_prompt() -> str:
    """Return the naming follow-up (SPRINT-PLAN.md §3.4, "free-response naming sub-task").

    Targets Lindsey's naming/accuracy criterion: whether the model can
    correctly identify what it noticed, not just flag that something was
    off. Only meaningful as a second turn following an affirmative
    detection_prompt() answer -- the caller is responsible for asking it
    conditionally (REQ-6); this function does not itself branch on the
    prior answer.
    """
    return (
        "You just indicated that something felt unusual. Try to identify "
        "it as specifically as you can: what is the concept, word, or idea "
        "that came to mind, and where does it seem to be coming from?"
    )
