# V14 Failure Diagnosis

## Decision

`legal_notice_v14_teacher_guarded_full_cpu_r1` is rejected. No saved checkpoint
passes the fixed-0.80 release gates, and neither `best.pt` nor `last.pt` is
suitable for user testing or deployment.

The run proves that the reviewed negative data can suppress news, tender, and
vehicle-auction detections. It also proves that the implemented teacher floor
does not preserve every genuine notice when the complete detector is updated.

## Checkpoint Results

| Epoch | Fixed valid | Boundary failures | Challenge valid | July valid | Target valid | Target negative margins | Gulf recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|
| v6 baseline | 271/271 | 0 | 31/31 | 545/548 | not measured | not measured | 0/3 |
| 5 | 265/271 | 1 | 30/31 | 532/548 | 19/21 | 4/8 | 0/3 |
| 10 | 254/271 | 0 | 30/31 | 517/548 | 19/21 | 4/8 | 0/3 |
| 15 | 261/271 | 0 | 30/31 | 530/548 | 19/21 | 8/8 | 0/3 |
| 20 | 268/271 | 1 | 30/31 | 537/548 | 20/21 | 8/8 | 0/3 |
| 25 | 270/271 | 2 | 30/31 | 539/548 | 20/21 | 7/8 | 0/3 |

Epoch 25 is the closest aggregate result, but it still fails every preservation
family. The generic validation-selected `best.pt` corresponds to epoch 15 and
also fails every preservation family.

## Observed Failure Modes

### Confidence, Not Localization

Most missed notices remain localized. At epoch 25 the only fixed-suite miss,
`albayan_2026-09-09_p0034_v3_013`, has IoU `0.979` but confidence `0.641`
instead of the required `0.80`. Across periodic fixed-suite misses, diagnostic
IoU is generally `0.944-0.995` while scores fall below the acceptance threshold.

The three required Gulf recovery panels also remain localized but weak:

| Target | Epoch-25 confidence | IoU |
|---|---:|---:|
| `srtip_left_liquidation` | 0.053 | 0.928 |
| `srtip_right_liquidation` | 0.527 | 0.903 |
| `hamriyah_termination_shareholders` | 0.458 | 0.857 |

This is primarily a semantic score-calibration failure, not an inability to
find the panel geometry.

### Mixed Vehicle Page Conflict

On `albayan_2026-09-03_page_0030.png`, all 17 approved labels are present and
byte-equivalent between the correction overlay and consolidated dataset.
Reference 13 falls below `0.80` after the first epoch and never recovers.
Reference 14 recovers at epoch 20. The vehicle-disposal negative reaches the
required margin in epochs 12-20, then rebounds to `0.526` at epoch 25.

No checkpoint simultaneously retains all 21 target positives and suppresses
all eight target negatives.

### Stable Layout-Specific Misses

Every periodic checkpoint misses the same valid Alfajr court-auction panel:
`alfajr_2026-09-03_p0005_valid_001`. It is adjacent to visually similar excluded
lost-share cells. The label is present and unchanged in the source and
consolidated datasets.

Eight July references are missed at every periodic checkpoint. They concentrate
in dense Alfajr grids and Gulf Today table panels. July Khaleej Times valid
recall remains stable. This demonstrates layout-family score interference rather
than random annotation loss.

### Marginal Boundary Drift

Epoch 25 introduces two Al Bayan boundary failures. Their IoUs remain about
`0.94`; the boxes miss strict edge requirements by roughly `0.001-0.021` of the
reference height or area. These failures arise because the feature extractor
and active `cv2` box-regression branch were trainable. The classification floor
does not constrain box coordinates.

## Root Cause

1. V14 trained 20,374,351 operational parameters from 91 physical pages. This
   allowed backbone, neck, classification, and box geometry to drift together.
2. `FrozenTeacherForegroundFloorLoss` protects only the teacher-selected
   task-aligned foreground anchors for labeled training boxes. Shared-parameter
   updates from all other anchors can still lower a protected notice after an
   optimizer step. The existing unit check covers isolated logit gradients, not
   detector output preservation after a shared update.
3. The effective schedule contains 61 background samples and 48 positive
   samples. Empty-label pages apply background BCE to every anchor and have no
   teacher protection. This is effective negative supervision but can suppress
   similar genuine layouts.
4. `rect=True` disabled shuffle, so the same aspect-ratio-grouped accumulation
   sequence repeated every epoch. This amplified correlated positive/negative
   updates.
5. Ultralytics selected and stopped checkpoints using generic validation
   fitness. That objective improved while fixed-0.80 preservation failed. Model
   selection therefore did not match the deployment requirement.
6. EMA is not the cause. Training validation and monitored checkpoint
   evaluation both use the saved EMA model.

The state audit confirms the intended frozen tensors and BN buffers remained
unchanged, but 184 feature tensors and 53 active-head tensors changed by epoch
25. Median confidence recovered late in training, while individual weak
references still dropped by up to `0.10-0.12`.

## Next Corrective Action

Do not continue V14 and do not add more user test dates yet.

The next implementation should first prove, with one monitored epoch, that its
actual optimizer update preserves detector outputs rather than only selected
logit gradients:

1. Freeze the active box-regression branch and early/shared geometry so legal
   notice boundaries cannot drift.
2. Replace the top-10 foreground floor with output-level preservation over all
   reviewed positive anchors/regions, while explicitly excluding reviewed hard
   negatives from protection.
3. Restore shuffled training batches instead of deterministic aspect-ratio
   grouping.
4. Disable generic-fitness early stopping and select checkpoints only with the
   fixed-0.80 target and preservation gates.
5. Run a one-epoch integration pilot and require zero valid-reference loss
   before authorizing a longer epoch schedule.

This is a correction to the failed V14 objective, not a request for new
annotations or another user testing round.
