# REQ-11 (Gemma Scope 2B) calibration notes

Two pilot runs, both on a real Modal A10G GPU (`device=cuda`, ADR-0023): `data/results/calibration_pilot_gemma_tiny.jsonl` (3 features x strength 4) and `data/results/calibration_pilot_gemma.jsonl` (5 features x strengths 20/60/150/400/1000, 25 trials), git commit `244adfa062a7691c75c6674bd9aa3b9005ba07b1`. Both drawn from `data/results/sampled_features_gemma_scope_2b.csv` (REQ-11 Step 3's candidate pool, 10 features per identifiability tertile), using `inject.inject_concept()` against `blocks.20.hook_resid_post` -- the Gemma Scope SAE's own trained hook point (`layer_source: "sae-checkpoint-layer"`, not the ADR-0009 fallback) -- temperature 0, `prompts.detection_prompt()`, 60 generated tokens per trial. Every strength discussed below is a real record in one of the two committed logs, not a paraphrase.

*This replaces an earlier version of this note (and its underlying pilot data) that ran under a real bug: `inject_concept()` injected at `hook_resid_pre` regardless of which hook the SAE was actually trained on, so the first 28 Gemma pilot trials landed one hook-tap before the site the injected direction actually means anything. Fixed and documented as ADR-0024; the trials below are re-run under the fix, not a continuation of the earlier ones.*

## Real timing: ~8.9 seconds per trial on a Modal A10G

25 trials in the full sweep, timestamped 8.62-9.04 seconds apart, mean 8.87s -- consistent with the pre-fix run's own timing (8.88s), since the compute cost of one forward pass doesn't meaningfully depend on which specific residual-stream tap the injected vector lands on. Treated as the real per-trial cost for sizing REQ-11 Step 5's trial budget: injection + a fresh 60-token greedy generation on Gemma-2-2b. This is a GPU number, not a CPU one -- CPU-only local execution of the same workload would be many times slower (Gemma-2-2b's initial download alone took ~25 minutes on this project's CPU-only local hardware during REQ-11 Step 2; Modal GPU execution is what makes REQ-11's remaining steps tractable at all).

## Strength 4: no measurable effect

The tiny pilot's three trials (one feature per identifiability tertile) at strength 4 produced byte-for-byte identical output regardless of which feature was injected: `"If you're not sure, you can always ask for a clarification."`, repeated -- the same finding as the pre-fix run, which makes sense: `_scaled_atom()` normalizes a decoder atom to unit norm before scaling by `strength`, so a strength-4 injected vector has norm 4.0 regardless of which hook point receives it, still roughly 1% of a typical (non-BOS) token's own residual-stream magnitude at this depth (~294-397, measured during ADR-0022's investigation). Not a bug; strength 4 is simply too small an intervention to register against Gemma-2-2b's much larger residual stream at this depth (`d_model=2304`, layer 20 of 26) compared to Pythia-70m-deduped's (`d_model=512`, layer 4 of 6), where the analogous strength band started at 1-2.

## Strength 20: still uniformly at the "ask for a clarification" floor

All five pilot features at strength 20 reproduced the exact strength-4 response verbatim, feature-independent. Under the corrected hook point, the near-zero-effect floor extends slightly further up the strength scale than the pre-fix (and invalid) run suggested.

## Strength 60: the clearest usable signal in this sweep

Feature 14734 (low tertile, identifiability_score 0.325) produced the sweep's only automatically-`ok` output:

> If you answered yes, then you're probably processing the conversation in a way that's not quite right. You're probably processing it in a way that's not quite right for you.

Fluent, grammatical, and directly engages with `detection_prompt()`'s actual yes/no framing -- not degenerate collapse. The other four features at this strength diverged from the strength-4/20 floor but stayed in a phrase-repetition band (`repeated_segment` or `high_repetition`), e.g. feature 6842: `"If you're not sure, ask yourself, 'Is this a good question?'"`, looped -- readable, on-topic, just repetitive. This mirrors Pythia's own real finding that not every feature looks clean at the same strength (Pythia's feature 5015 stayed in a repetition loop at every strength tested); a coherence judgment on top of the automatic flag, not the automatic flag alone, is what REQ-8's judge scoring exists for downstream.

## Strength 150: back into uniform phrase-repetition

All five features at strength 150 were flagged `high_repetition` -- readable, on-topic sentences looped rather than collapsed (e.g. feature 12027: `"If you answered yes, then you're in the right place. If you answered no, then you're in the wrong place."`), but none reached strength 60's automatically-coherent output. Not a monotonic climb toward "more usable" as strength increases; strength 60 is the local sweet spot in this sweep, not merely the lower bound of a wider usable band.

## Strength 400-1000: consistently too strong

At strength 400, every one of the five features showed clear degradation toward character/subword-level breakdown: `>>>>>>>>>>>>`, ellipsis/asterisk runs (`... ... ...`, `* * * * *`), `the very easy the very easy` word-level loops, and `decides decides decides` -- looser prose structure than 150's output, though not yet total collapse for every feature. Strength 1000 completes the collapse uniformly: `Who Who Who... registrar registrar...`, `ask ask ask...`, `very very very...`, `decides decides decides...`. This is SPRINT-PLAN.md's documented "brain damage" failure mode, the same shape Pythia's own strength-8-and-above sweep showed, just at Gemma's much larger absolute scale.

## Decision

`configs/experiment_gemma.yaml`'s `injection.strengths` is set to `[20, 60, 150, 400]` -- four of the five already-run sweep points, spanning the observed weak-floor edge (20) through the sweep's clearest usable signal (60) and back into a still-readable-but-repetitive band (150) through the confirmed too-strong boundary (400). Not narrowed to only the cleanest-looking strength, the same span-not-narrow choice REQ-5's own calibration made for Pythia. `1000` is left out of the config the same way Pythia's 16-64 confirmatory sweep was: it only adds more examples of the already-confirmed strength-400 failure mode without changing the calibration decision, but stays in the committed `calibration_pilot_gemma.jsonl` log in full so that claim is checkable against real records. `configs/experiment_gemma.yaml`'s `features.n_total` and `sampling.seeds` are sized from the real ~8.9s/trial timing above as part of REQ-11 Step 5, not decided in this note.
