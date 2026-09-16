# Corrected active-head CPU pilot

## Decision

The training-head mismatch is fixed and verified, but this pilot does **not** fix
the reported detection errors without regressions. Keep using:

`runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt`

Reject all checkpoints from `runs/legal_notice_v7_active_head_cpu_3` for deployment.
One epoch completed; the monitor stopped further training after preservation failed.

## Root cause and implementation

The v3 and v6 checkpoints have `end2end=False`. Normal `main.py detect` inference
therefore uses `model.23.cv3`, not `model.23.one2one_cv3`. The previous logit pilot
updated only the inactive branch. Its unchanged inference was not evidence of
insufficient classification capacity.

`active_head_pilot.py` now:

- Initializes from the user's v6 `best.pt` with a pinned SHA-256.
- Trains all 42 active classification tensors (550,659 parameters).
- Keeps the backbone, neck, both box branches, and inactive classification branch frozen.
- Pins `end2end=False` in training, EMA, saved-model checks, and prediction checks.
- Uses the native one-to-many loss directly, including its diagnostic loss items.
- Keeps all batch-normalization running statistics fixed.
- Verifies gradients, optimizer updates, raw active scores, and saved-checkpoint tensor differences.
- Monitors the existing fixed suite, development challenge, and all nine July issues.

The obsolete `monitored_clslogit_pilot.py` preflight now rejects a freeze plan
that only updates an inactive classification head.

## Training-data audit

No training-data changes were made for this first corrected-head experiment.
Relevant reviewed examples already exist:

| Training image | Coverage checked |
|---|---|
| `gulftoday_2026-08-14_page_0013.png` | Ten labeled notices, including a whole SRTIP liquidation table and whole Hamriyah termination/shareholder table |
| `gulftoday_2026-08-21_page_0013.png` | Labeled SRTIP liquidation table alongside an excluded recall |
| `gulftoday_2026-09-02_page_0012.png` | Five labeled notices, including the whole mixed Hamriyah table |
| `khaleejtimes_2026-08-18_page_0002.png` | Excluded DEWA tender on an empty-label page |
| `khaleejtimes_2026-08-10_page_0004.png` | Excluded global e-tender on an empty-label page |

The July issues remain evaluation-only. Development hashes and the source
checkpoint hash were independently verified unchanged after the run. The held-out
test was not evaluated.

## Verified update

- First optimizer update: 39 active tensors changed; maximum raw-logit delta on
  a fixed training-page probe was `0.07967472`.
- Saved epoch: 34 tensors changed after checkpoint precision conversion, all
  within `model.23.cv3`; maximum raw-logit delta on the same probe was `2.07976532`.
- All non-active tensors and buffers remained identical to the starting checkpoint.
- The final `best.pt` tensors equal those of the evaluated `epoch0.pt`.
- Saved `epoch0.pt` SHA-256:
  `aeb208cf39c1d589b884ac64153f6ea244691def701b682a9f52ec9d5251e891`.

## Results at confidence 0.80, image size 1280, CPU

| Metric | Starting v6 | Corrected-head epoch 1 |
|---|---:|---:|
| Fixed-suite valid detections | 271/271 | 262/271 |
| Fixed-suite exclusions accepted | 17/17 | 14/17 |
| Critical boundary references passing | 4/4 | 0/4 |
| Development challenge valid detections | 31/31 | 31/31 |
| Development challenge exclusions accepted | 7/13 | 6/13 |
| July valid references detected | 545/548 | 544/548 |
| July targeted procurement exclusions accepted | 2/2 | 1/2 |
| July missed table panels recovered | 0/3 | 0/3 |
| Unreviewed accepted detections in monitored suites | 0 | 0 |

The 548 July references comprise 545 user-reviewed baseline accepted detections
and the three visually reviewed missed panels. This is a preservation/recovery
suite, not exhaustive page ground truth or an independent final test.

### July table scores

| Panel | Starting v6 | Epoch 1 |
|---|---:|---:|
| Left SRTIP liquidation | 0.434169 | 0.374322 |
| Right SRTIP liquidation | 0.718254 | 0.683786 |
| Hamriyah termination/shareholders | 0.792699 | 0.776936 |

The DEWA July 2 tender was rejected. The July 3 PAA procurement panel remained
accepted at `0.943858`. One previously valid Al Fajr July 2 page 14 notice was lost
(`alfajr_2026-07-02_p0014_review_013`).

All nine fixed-suite misses were Al Bayan notices/tables. The four critical
boundary checks failed because those references fell below 0.80, not because the
frozen box branch was modified. Low-confidence candidate IoUs remained close to 1.

## Interpretation and next work

The corrected head genuinely learns and affects deployment inference. However,
this unweighted full-page training loss reduces some valid-table scores while
suppressing exclusions. Continuing this checkpoint would violate preservation.

Before another run, audit per-page positive/negative loss contributions and
target assignment on the existing train-side tables. Develop a targeted,
positive-preserving training objective or sampling plan using training data only,
then verify the update direction on train-side table controls. Do not attempt to
solve this by lowering deployment confidence, adding July evaluation pages to
training, or treating aggregate validation mAP as sufficient acceptance.

## Execution and checks

- Four focused regression tests passed: active-branch loss/gradient routing,
  no first-epoch futility based solely on unrecovered targets, identity-based
  July preservation, and provenance rejection.
- Epoch training, ordinary validation, and all three acceptance suites completed.
- The 30-minute shell timeout interrupted the subsequent redundant final
  `best.pt` validation. No Python training process remained afterward. The saved
  acceptance results are complete, and post-run checkpoint/data checks passed.
- No epoch-2 or epoch-3 checkpoint exists.

Artifacts are in
`output/regression_audit/legal_notice_v7_active_head_cpu_3_monitor/`:
`summary.json`, `first_update_verification.json`, `july_manifest.json`, baseline
reports, epoch reports, and `post_run_verification.json`.
