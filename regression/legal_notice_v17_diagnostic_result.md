# V17 Complete-Contract Diagnostic Result

## Decision

V17 is rejected at its precommitted diagnostic gate. Do not run the conditional
40-proposal fit, do not run candidate release evaluation, and do not deploy or
promote the diagnostic checkpoint.

The retained development baseline remains R7:

- Checkpoint: `runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt`
- SHA256: `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`

The rejected V17 diagnostic is:

- Checkpoint: `runs/legal_notice_v17_guarded_trust_region_cpu_r1/weights/diagnostic.pt`
- SHA256: `be3f556653e183b17f750a46dcbdeae8863a617450cee853867dcc253241699c`

## Complete Contract

The unified evaluator executed the complete release inventory against R7 before
fitting. Its summary is
`output/regression_audit/legal_notice_v17_r7_complete_baseline/release_summary.json`.
R7 preserved all 271 fixed, 31 challenge, 548 July, 21 target, and 400
historical valid references. It remained ineligible because it accepted reviewed
exclusions and missed four additional recovery references.

The V17 correction inventory was assembled from the same release manifests:

- 382 unique, completely reviewed pages
- 1,294 valid references
- 1,290 starting preservation references
- 4 starting recovery references
- 45 explicit reviewed negative regions
- 382 complete-page background guards

The pinned cache manifest is
`output/regression_audit/legal_notice_v17_guarded_trust_region_cpu_r1/train_cache/cache_manifest.json`
with SHA256
`b48e9db79ddd28c42cf4dac7e7a020d78b84159fa8e48a70be0840e6450fc014`.

Native-versus-cached output equivalence passed for every distinct original/input
shape represented in the inventory. Maximum confidence drift was
`1.7881393432617188e-7`, minimum matched IoU was `0.9999994039535522`, and
the maximum native-versus-cached head-output delta was zero.

## Feasibility Preflight

The preflight found a common first-order descent direction across 44 active
constraints:

- 4 recovery constraints
- 39 active explicit-rejection constraints
- 1 near-threshold preservation guard
- Minimum pairwise cosine: `-0.937237250005674`
- Minimum common-direction improvement: `0.008492656882154198`

Six reviewed negative regions already met the correction ceiling and therefore
had inactive rejection hinges. Complete preservation and background safety were
still checked transactionally after every proposal.

Evidence:
`output/regression_audit/legal_notice_v17_guarded_trust_region_cpu_r1/gradient_feasibility.json`.

## Diagnostic Outcome

The bounded trust-region diagnostic used 12 proposals and 1,772.47 seconds of
recorded model compute. Five updates were accepted and seven were rolled back.
Every rejection was caused by a valid-reference preservation regression at the
proposed radius.

Safe accepted updates improved every recovery and did not worsen any reviewed
negative region, but they did not make enough progress:

| Gate metric | Initial | Final | Reduction | Required |
| --- | ---: | ---: | ---: | ---: |
| Recovery shortfall | 0.293843 | 0.283301 | 3.588% | 10% |
| Explicit negative excess | 15.252983 | 15.214930 | 0.249% | 10% |

Additional terminal facts:

- Accepted updates: `5/12`
- Recoveries crossing deployment confidence: `0/4`
- Negative regions at the `0.49` correction margin: `6/45`
- Preservation failures after accepted updates: `0`
- New accepted background detections: `0`
- Background confidence increases: `0`
- Every recovery improved: passed
- Every explicit negative did not worsen: passed
- Serialized checkpoint prediction equivalence: passed with zero confidence drift
- Frozen parameter and buffer checks: passed after every proposal and after reload

Authoritative progress and gate evidence:
`output/regression_audit/legal_notice_v17_guarded_trust_region_cpu_r1/fit_progress.json`.

## Interpretation

The complete-contract result is stronger than V16's partial-inventory result.
A shared local correction direction exists, but useful step sizes collide with
strict preservation, while preservation-safe steps move the 39 active negative
constraints too slowly. Continuing the same frozen `model.23.cv3` correction
for 40 more proposals would violate the precommitted continuation rule and is
not authorized by the evidence.

Any next model attempt must change the optimization parameterization or
trainable representation while retaining this complete evaluator and all V17
safety guards. More proposals with the rejected V17 formulation are not a
valid next step.
