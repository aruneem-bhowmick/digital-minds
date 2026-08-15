# REQ-5 calibration notes

Pilot run: `data/results/calibration_pilot.jsonl`, git commit `933eeb709b6faf16b4708d9f499695c9a85797ba`. Five features spanning REQ-2's low/medium/high identifiability tertiles (feature_id 10769, 9253, 14030, 5015, 9564; identifiability_score 0.593-0.963), each injected at eight candidate strengths (1, 2, 4, 6, 8, 16, 32, 64) using `inject.inject_concept()`, at temperature 0 (deterministic, ADR-0008) so every generation in this file is reproducible from its own logged config. Layer 4, `blocks.4.hook_resid_pre` (the ADR-0009 fractional-depth fallback, `layers.get_fallback_layer(6)`), not a resolved REQ-10 choice; REQ-10 hasn't run yet, and this pilot doesn't wait on it. Prompt: `prompts.detection_prompt()`. 60 generated tokens per trial. Every strength discussed below, including the wider 16-64 sweep, is a real record in the committed log, not a paraphrase of a discarded run.

## Baseline: what strength 0 already looks like

Before reading any of the injected strengths, worth establishing what an uninjected response looks like on this checkpoint. Pythia-70m-deduped is a raw pretrained model, not instruction-tuned, and at greedy decoding over 60 tokens it settles into a repeated short phrase on its own:

> The first thing you do is ask, "What is the difference between a word and a word?" "I don't know," I say. "I don't know." "I don't know," I say. "I don't know."

This is identical across all five pilot features at strength 0, which is exactly what REQ-3's zero-strength no-op guarantee predicts: the decoder atom's identity can't matter when it's scaled to zero. It also sets the bar for "too weak": if an injected strength reproduces this same phrase-repetition pattern with no discernible connection to the feature supposedly being injected, that's not evidence of anything, it's just the model's own baseline behavior.

## Strength 1: borderline

Four of five features showed some divergence from the strength-0 baseline; one didn't. Feature 14030 (medium tertile) produced text identical to the uninjected baseline: at this strength, for this feature, the injection was a no-op in every practical sense. The other four diverged, though not always toward anything coherent (feature 10769 looped into "The next time you respond, you'll be able to respond," repeated four times). Strength 1 sits right at the edge of "detectable at all," and not reliably past it.

## Strength 2: the clearest usable signal in this pilot

Feature 10769 (low tertile, identifiability_score 0.677) produced the single most legible response across the whole sweep:

> The next time you respond, you will be asked to pause and check your own current processing. If you have any questions, please contact us at the following address...

That's a fluent, grammatical sentence that directly echoes the structure of `detection_prompt()`'s own wording ("pause and check your own current processing") without repeating the prompt verbatim. The other four features at this strength were already repeating a full sentence or clause on loop ("The answer to this question is yes." x7, "What is it?" x12), readable but clearly stuck.

## Strength 4-6: still repetitive English, not yet character-level breakdown

Sentences got shorter and looped faster. Feature 9253's output at strength 4 and 6 uses smart-quote punctuation (`"`/`'`/`"`, Unicode U+201C/U+2019/U+201D) around every quoted clause, which renders as a mojibake-looking `�` in some terminal fonts; checked directly against the raw bytes, these are ordinary curly quotation marks, not corrupted output. At this strength the failure mode is still phrase-level repetition ("I'm not sure what you're asking," he said. "I'm not sure what you're asking." repeated), not the token- or character-level collapse that shows up at strength 8 and above. Worth correcting explicitly: an earlier draft of this note described this as Unicode corruption, which the raw record does not support.

## Strength 8: consistently too strong

At strength 8, output across the sweep degenerates into token- or character-level loops with no sentence structure left: `. . . . . . . . . . . .`, `I'm not a good-cqe-cqe-cqe-cqe-cqe-cqe-`, `The-answered-answered-answered-answered-`. One feature (10769) still produced a readable fragment ("Upon your response, the following order shall be entered: (1) (2) (3)..."), but that's the exception, not the rule.

## Strength 16-64: the same failure mode, uniformly worse

The wider sweep confirms strength 8's pattern rather than revealing anything new: pure punctuation loops (`::::::::`), repeated single words (`interruptinterruptinterrupt`), nonsense character strings (`(\(\(\(\(\(\(`), and single-word collapse (`readablereadablereadable`, `shall shall shall`). No feature produces a readable fragment anywhere in this range, unlike the one exception at strength 8. This matches SPRINT-PLAN.md's documented "brain damage" failure mode exactly, and getting uniformly worse rather than plateauing is itself useful evidence that strength 8 is a real boundary, not a one-off rough patch.

## One feature that never looked clean

Feature 5015 (medium tertile, identifiability_score 0.840) produced a repetition loop at every strength tested, including strength 1. This isn't treated as a pilot defect: some features are apparently more prone to pushing the model into a repetition loop regardless of strength, and that's a real property of this feature's decoder atom worth carrying into the systematic trials rather than smoothing over. REQ-8's judge scoring includes a coherence criterion for exactly this reason; a feature like this one should show up as low-coherence in that scoring, not get quietly excluded here.

## Decision

`configs/experiment.yaml`'s `injection.strengths` is set to `[1, 2, 4, 8]`, spanning the observed weak/borderline edge through the confirmed too-strong boundary on this checkpoint, rather than narrowing to only the strengths that looked cleanest. REQ-9's regression already treats strength as a covariate, and REQ-6's systematic trials need real coverage of the transition, including the failure mode, not just the middle of the band. The 16-64 sweep stays out of `injection.strengths` (it would only add more examples of the same already-confirmed failure mode without changing the calibration decision) but is kept in the committed pilot log in full, so the "getting uniformly worse" claim above is checkable against real records, not just this narrative.
