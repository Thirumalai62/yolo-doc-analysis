# Legal Notice V16 Guarded Diagnostic Result

## Decision

V16 is stopped at its precommitted diagnostic gate. The additional 100-proposal
fit and full release evaluation are not authorized. Neither V16 diagnostic
checkpoint is eligible for promotion; retain the R7 checkpoint as the active
development baseline.

## Pinned Artifacts

- Active baseline: `runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt`
- Active baseline SHA256: `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`
- Prepared inventory: 96 unique pages, 352 valid notices, 8 explicit negative regions
- Cache manifest SHA256: `52342cb7baa6c5e59380cb0e08ec772c265daf8792584ca3e8487a1bc4d0101e`
- Rejected aggregate diagnostic SHA256: `51fb18833a7b17f7cbf4823ace893e0b79efc0440a4cde49e807a1bffc94d45b`
- Rejected directional diagnostic SHA256: `f1e98d1eb158f63ea530196a67071a25e2341d7ab9b85d4e15b95645c0df7377`
- Directional evidence: `output/regression_audit/legal_notice_v16_guarded_directional_cpu_r1/fit_progress.json`

## Preparation Result

The cached classification path reproduced ordinary R7 prediction with maximum
confidence difference `2.4e-7`, minimum matched IoU `0.999999`, and zero cached
head-output drift. The immutable training inventory assigned:

- 348 preservation references
- 4 actual recovery references
- 8 explicit rejection regions
- 96 complete-page background guards

The restored replay and guard images passed their recorded hashes. R7 retained
its previously verified 271/271 fixed, 31/31 challenge, 548/548 July, 3/3 Gulf
recovery, and 4/4 critical-boundary results.

## Aggregate Diagnostic

The original globally normalized Adam direction was proposed ten times while
the learning rate was reduced from `1e-4` to below `1e-7`. Every proposal was
rolled back. No model update was accepted.

The active correction objectives conflicted with the safety guard:

- Recovery versus background gradient cosine: `-0.822`
- The marginal September 9 valid notice fell below its exact baseline floor.
- One September 15 recovery and one target exclusion moved in the wrong
  direction even at the smallest tested step.

No R7 state was changed by these rejected transactions.

## Directional Diagnostic

Individual-gradient analysis covered 19 active constraints: four recovery,
eight explicit rejection, six background, and one marginal preservation guard.
Pairwise cosine reached `-0.885`, but a common first-order descent direction was
feasible with minimum directional improvement `0.0201`.

The remaining ten diagnostic proposals used that deterministic minimum-norm
direction. All ten were accepted:

- Preservation failures: 0
- New accepted background detections: 0
- Background confidence increases: 0
- Every recovery reference improved
- Every explicit negative region decreased
- Changed tensors: 26, all under the operational `model.23.cv3.*` scope
- FP32 checkpoint save/reload identity: passed

The measured progress was nevertheless below the gate:

| Metric | Required | Achieved |
| --- | ---: | ---: |
| Recovery shortfall reduction | >=10% | 2.174% |
| Explicit-negative excess reduction | >=10% | 0.307% |

No recovery reached deployment confidence `0.80`, and no explicit negative
reached the required `0.49` margin during the diagnostic budget.

## Case Movement

All four recovery diagnostics increased:

| Reference | R7 | V16 diagnostic |
| --- | ---: | ---: |
| `albayan_2026-09-03:p0029:label_4` | 0.796240 | 0.797774 |
| `khaleejtimes_2026-08-17:p0014:label_2` | 0.588592 | 0.590226 |
| `gulftoday_2026-09-15:p0013:label_10` | 0.781696 | 0.783990 |
| `gulftoday_2026-09-15:p0013:label_11` | 0.722621 | 0.724352 |

All eight target exclusions decreased, but only by approximately `0.0004` to
`0.0020`; their final raw confidences remained between `0.6854` and `0.9644`.

## Closure

The diagnostic established that safe simultaneous movement is possible within
R7's frozen representation, but the pinned step policy does not make enough
progress to justify the longer fit. Continuing because the direction looks
promising would violate the agreed gate and repeat the previous pattern of
extending unsuccessful runs.

No full regression bundle, historical holdout, or production-equivalence test
was consumed after the gate failure. R7 remains unchanged and is the only
retained checkpoint from this development sequence.
