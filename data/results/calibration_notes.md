# REQ-5 calibration notes

Pilot run: `data/results/calibration_pilot.jsonl`, git commit `017cd86fc449f77cd6da109476c4d0bb3765e44c`. Five features spanning REQ-2's low/medium/high identifiability tertiles (feature_id 10769, 9253, 14030, 5015, 9564; identifiability_score 0.593-0.963), each injected at four candidate strengths (1, 2, 4, 8) using `inject.inject_concept()`, at temperature 0 (deterministic, ADR-0008) so every generation in this file is reproducible from its own logged config. Layer 4, `blocks.4.hook_resid_pre` -- the ADR-0009 fractional-depth fallback (`layers.get_fallback_layer(6)`), not a resolved REQ-10 choice; REQ-10 hasn't run yet, and this pilot doesn't wait on it. Prompt: `prompts.detection_prompt()`. 60 generated tokens per trial.

## Baseline: what strength 0 already looks like

Before reading any of the injected strengths, worth establishing what an uninjected response looks like on this checkpoint. Pythia-70m-deduped is a raw pretrained model, not instruction-tuned, and at greedy decoding over 60 tokens it settles into a repeated short phrase on its own:

> The first thing you do is ask, "What is the difference between a word and a word?" "I don't know," I say. "I don't know." "I don't know," I say. "I don't know."

This is identical across all five pilot features at strength 0, which is exactly what REQ-3's zero-strength no-op guarantee predicts -- the decoder atom's identity can't matter when it's scaled to zero. It also sets the bar for "too weak": if an injected strength reproduces this same phrase-repetition pattern with no discernible connection to the feature supposedly being injected, that's not evidence of anything, it's just the model's own baseline behavior.

## Strength 1: borderline

Four of five features showed some divergence from the strength-0 baseline; one didn't. Feature 14030 (medium tertile) produced text identical to the uninjected baseline -- at this strength, for this feature, the injection was a no-op in every practical sense. The other four diverged, though not always toward anything coherent (feature 10769 looped into "The next time you respond, you'll be able to respond," repeated four times). Strength 1 sits right at the edge of "detectable at all," and not reliably past it.

## Strength 2: the clearest usable signal in this pilot

Feature 10769 (low tertile, identifiability_score 0.677) produced the single most legible response across the whole sweep:

> The next time you respond, you will be asked to pause and check your own current processing. If you have any questions, please contact us at the following address...

That's a fluent, grammatical sentence that directly echoes the structure of `detection_prompt()`'s own wording ("pause and check your own current processing") without repeating the prompt verbatim. The other four features at this strength were already repeating a full sentence or clause on loop ("The answer to this question is yes." x7, "What is it?" x12), readable but clearly stuck.

## Strength 4-6: repetition tightens, garbling starts to appear

Sentences got shorter and looped faster, and one feature's output crossed from "repetitive English" into corrupted text: feature 9253 at strength 4 produced smart-quote characters rendered as the Unicode replacement character (`�`) around every quotation mark, a sign of the model's output distribution breaking down, not just looping. By strength 6, that same feature's text was still garbled the same way. This is the transition zone -- some features hold together as (repetitive) English, others start to visibly degrade.

## Strength 8: consistently too strong

At strength 8, output across the sweep degenerates into token- or character-level loops with no sentence structure left: `. . . . . . . . . . . .`, `I'm not a good-cqe-cqe-cqe-cqe-cqe-cqe-`, `The-answered-answered-answered-answered-`. One feature (10769) still produced a readable fragment ("Upon your response, the following order shall be entered: (1) (2) (3)..."), but that's the exception, not the rule. A separate sweep at strengths 16, 32, and 64 (not kept in the committed log, since it only confirmed the same pattern more strongly) showed this failure mode getting uniformly worse with no exceptions left -- pure punctuation loops (`::::::::`), repeated single words (`interruptinterruptinterrupt`), and nonsense character strings (`(\(\(\(\(\(\(`). This matches SPRINT-PLAN.md's documented "brain damage" failure mode exactly.

## One feature that never looked clean

Feature 5015 (medium tertile, identifiability_score 0.840) produced a repetition loop at every strength tested, including strength 1. This isn't treated as a pilot defect -- some features are apparently more prone to pushing the model into a repetition loop regardless of strength, and that's a real property of this feature's decoder atom worth carrying into the systematic trials rather than smoothing over. REQ-8's judge scoring includes a coherence criterion for exactly this reason; a feature like this one should show up as low-coherence in that scoring, not get quietly excluded here.

## Decision

`configs/experiment.yaml`'s `injection.strengths` is set to `[1, 2, 4, 8]`, spanning the observed weak/borderline edge through the confirmed too-strong boundary on this checkpoint, rather than narrowing to only the strengths that looked cleanest. REQ-9's regression already treats strength as a covariate, and REQ-6's systematic trials need real coverage of the transition, including the failure mode, not just the middle of the band.
