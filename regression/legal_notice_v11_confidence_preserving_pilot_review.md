# V11 Confidence-Preserving Pilot Review

## Status

The one authorized epoch completed. Its checkpoint was rejected and authorization is revoked; rerunning this configuration is prohibited.

## R2 Finding

R2 did not lose localization. All 271 fixed valid references retained localized candidates above 0.05, but median confidence fell from about 0.9666 to 0.7999. The standard validation mAP remained high while fixed, challenge, July, Gulf recovery, and targeted-valid gates failed. The native soft classification targets therefore applied broad downward pressure that the 61 background-only samples amplified.

## Pilot Change

The pilot keeps native task-aligned assignment and native background BCE. On anchors that the frozen starting model assigns to labeled foreground, its starting probability becomes a floor for the soft classification target:

`protected_target = max(native_target, baseline_probability)`

At initialization, the protected-anchor logit gradient cannot lower the baseline score. Unassigned anchors, including target negatives, retain the native BCE-to-zero gradient.

The real-model preflight on `gulftoday_2026-08-14_page_0013.png` found 100 foreground anchors, protected 75 logits whose native targets were lower than the baseline probabilities, and verified finite gradients in all 42 selected tensors.

## Scope And Optimizer

- One epoch over the unchanged 109-sample v11 schedule.
- Only `model.23.cv3.*` is trainable: 42 tensors and 550,659 parameters.
- Features, box head, one-to-one branches, and all BN running statistics remain frozen.
- AdamW learning rate is `1e-6`, 50 times lower than the rejected R2 head rate.
- Weight decay and warmup are zero.
- Effective batch is 8; training uses rectangular, deployment-like geometry.
- Historical test remains locked.

## Mandatory Gates

- Fixed: all 271 valid references, all four critical boundaries, and no exclusion regression.
- Challenge: all 31 valid references and no exclusion regression.
- July: all 548 prior valid references and all three Gulf recovery targets.
- Target: all 21 valid references and all eight negative regions at or below 0.50.
- Every suite must have median valid-confidence drop at most 0.01.
- Every paired valid reference must have confidence drop at most 0.05.
- Only the 42 configured `cv3` tensors may change.
- Any preservation failure rejects the checkpoint.

## Storage Lifecycle

Preflight validates source PDFs and pinned page hashes without rendering. Evaluation renders one issue at a time, evaluates it, and cleans only files recorded as newly generated. A real 36-page cycle used 212,919,190 bytes and reclaimed all of them; this is the largest measured issue and replaces the previous 1,866,332,374-byte all-suite peak.

## Approval Decision

Rejected. The checkpoint preserved 270 of 271 fixed references and passed 0 of 8 target negative margins. Keep the v8 R7 baseline deployed and do not rerun this configuration.
