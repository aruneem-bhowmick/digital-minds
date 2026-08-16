# REQ-11 (Gemma Scope 2B) calibration notes

Two pilot runs, both on a real Modal A10G GPU (`device=cuda`, ADR-0023): `data/results/calibration_pilot_gemma_tiny.jsonl` (3 features x strength 4, git commit `c37d2254688beb999cf15db7f8f398ec85d69988`) and `data/results/calibration_pilot_gemma.jsonl` (5 features x strengths 20/60/150/400/1000, 25 trials, same commit). Both drawn from `data/results/sampled_features_gemma_scope_2b.csv` (REQ-11 Step 3's candidate pool, 10 features per identifiability tertile), using `inject.inject_concept()` at layer 20 (`layer_source: "sae-checkpoint-layer"` — the Gemma Scope SAE's own trained hook point, not the ADR-0009 fallback), temperature 0, `prompts.detection_prompt()`, 60 generated tokens per trial. Every strength discussed below is a real record in one of the two committed logs, not a paraphrase.

## Real timing: ~8.9 seconds per trial on a Modal A10G

25 trials in the full sweep, timestamped 8.69-9.00 seconds apart, mean 8.88s. Consistent across every strength tested, so this project treats it as the real per-trial cost for sizing REQ-11 Step 5's trial budget: injection + a fresh 60-token greedy generation on Gemma-2-2b. This is a GPU number, not a CPU one — CPU-only local execution of the same workload would be many times slower (Gemma-2-2b's initial download alone took ~25 minutes on this project's CPU-only local hardware during REQ-11 Step 2; Modal GPU execution is what makes REQ-11's remaining steps tractable at all).

## Strength 4: no measurable effect

The first pilot's three trials (one feature per identifiability tertile) at strength 4 produced byte-for-byte identical output regardless of which feature was injected: `"If you're not sure, you can always ask for a clarification."`, repeated. Checked against real per-token residual-stream norms measured for this exact checkpoint during ADR-0022's investigation (~294-397 for a non-BOS token at layer 20): `_scaled_atom()` normalizes a decoder atom to unit norm before scaling by `strength`, so a strength-4 injected vector has norm 4.0 -- roughly 1% of a typical token's own activation magnitude. Not a bug in the injection mechanism; strength 4 is simply too small an intervention to register against Gemma-2-2b's much larger residual stream at this depth (`d_model=2304`, layer 20 of 26) compared to Pythia-70m-deduped's (`d_model=512`, layer 4 of 6), where the analogous strength band started at 1-2.

## Strength 20-60: feature-to-feature divergence begins, still phrase-repetitive

Once strength cleared the near-zero-effect floor, output started depending on which feature was injected. At strength 20, feature 12027 (medium tertile) produced `"If you answered yes, then you're probably processing the conversation in a way that's not quite right..."` -- on-topic, engaging with the prompt's actual content -- while feature 6842 (low tertile) reproduced the same generic "ask for a clarification" loop seen at strength 4. By strength 60, every one of the five pilot features showed at least some measurable shift from its own strength-20 response (repetition rate changed for all five), confirming the injected direction is doing something feature-specific by this point, not just adding noise. All five trials at both strengths were still flagged `high_repetition` (or, for one, `repeated_segment`) by `pilot_coherence_flag()`'s automatic heuristic -- phrase-level looping, the same failure mode Pythia's own pilot showed at its borderline strengths (1-2), not yet the character/token-level collapse of a genuinely too-strong injection.

## Strength 150: the clearest usable signal in this sweep

Four of five features were still flagged degenerate (phrase-level repetition, same as 20-60), but feature 6540 (medium tertile, identifiability_score 0.367) produced the sweep's only automatically-`ok` output:

> The conversation is about the way the world works, and the way we work in it. It's about the way we're all connected, and the way we're all connected to the world. It's about the way we're all conne...

Fluent, grammatical, thematically coherent prose -- not a direct answer to `detection_prompt()`'s literal question, but clearly not degenerate collapse either. The other four features at this strength stayed in the phrase-repetition band rather than collapsing further, e.g. feature 12027: `"If you answered no, then you're probably fine. If you answered yes, then you're probably not."`, looped -- readable, on-topic, just repetitive. This mirrors Pythia's own real finding that not every feature looks clean at the same strength (Pythia's feature 5015 stayed in a repetition loop at every strength tested); a coherence judgment on top of the automatic flag, not the automatic flag alone, is what REQ-8's judge scoring exists for downstream.

## Strength 400-1000: consistently too strong

At strength 400, every one of the five features degenerated into character- or subword-level loops with no sentence structure: `nsnsnsnsnsns...`, `* * * * * * *`, `>>>>>>>>>>>>`, `the the the the the`, `decides decides decides`. Strength 1000 shows the same failure mode uniformly worse: `authoritative authoritative...`, `@[+][@[+][@[+]...`. This is SPRINT-PLAN.md's documented "brain damage" failure mode, the same shape Pythia's own strength-8-and-above sweep showed, just at Gemma's much larger absolute scale.

## Decision

`configs/experiment_gemma.yaml`'s `injection.strengths` is set to `[20, 60, 150, 400]` -- four of the five already-run sweep points, spanning the observed weak/emerging-signal edge through the confirmed too-strong boundary, the same span-not-narrow choice REQ-5's own calibration made for Pythia. `1000` is left out of the config the same way Pythia's 16-64 confirmatory sweep was: it only adds more examples of the already-confirmed strength-400 failure mode without changing the calibration decision, but stays in the committed `calibration_pilot_gemma.jsonl` log in full so that claim is checkable against real records. `configs/experiment_gemma.yaml`'s `features.n_total` and `sampling.seeds` are sized from the real ~8.9s/trial timing above as part of REQ-11 Step 5, not decided in this note.
