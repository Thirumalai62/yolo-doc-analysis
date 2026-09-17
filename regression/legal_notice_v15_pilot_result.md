# Legal Notice V15 Reviewed-Classification Pilot Result

## Decision

The one-epoch V15 pilot is rejected and is not authorized to continue. Do not
promote `runs/legal_notice_v15_reviewed_cls_cpu_r4/weights/epoch0.pt`.

The checkpoint preserved the reviewed positive notices and critical boundaries,
and it moved every target negative region in the intended direction. However,
it moved every explicit recovery notice in the wrong direction. This fails the
pilot requirement for demonstrated recovery-objective learning, so additional
epochs are not justified under the bounded training plan.

## Reproducibility

- Baseline: `runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt`
- Baseline SHA256: `3204479f9feceab7a3124f49a2a1bde2a9ae13425e473b207dde524f04c98283`
- Candidate: `runs/legal_notice_v15_reviewed_cls_cpu_r4/weights/epoch0.pt`
- Candidate SHA256: `70f7ec3a132e6f9fff8f1827d8746c0c2a0daf58d47e0d0e77376ca1ddc4f3b2`
- Acceptance inventory SHA256: `b2ba84fa560e3702707e8dd41503cab2fc909ec79bcda9e26e42b355c5e5b3bd`
- Authoritative monitor: `output/regression_audit/legal_notice_v15_reviewed_cls_cpu_r4_monitor/monitor_summary.json`
- Baseline comparison: `output/regression_audit/legal_notice_v15_reviewed_cls_cpu_r4_monitor/pilot_diagnosis.json`

## One-Epoch Outcome

At deployment confidence `0.80`:

| Suite | Valid accepted | Exclusions accepted | Critical boundaries |
| --- | ---: | ---: | ---: |
| Fixed | 271/271 | 17 | 4/4 |
| Challenge | 31/31 | 7 | n/a |
| July | 545/548 | 2 | n/a |
| Target | 21/21 | 10 detections in 8 regions | n/a |

No reviewed target negative reached the required rejection margin. The
candidate reduced the maximum confidence in all eight regions by between
`0.000482` and `0.003397`; one marginal baseline detection moved from
`0.802372` to `0.799569`. The overall target exclusion count consequently fell
from 11 to 10 accepted detections.

The movement was not selective. All four training recovery references lost
confidence:

| Recovery reference | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `gulftoday_2026-08-14_page_0013.png:label_7` | 0.639965 | 0.634749 | -0.005216 |
| `gulftoday_2026-08-14_page_0013.png:label_10` | 0.878938 | 0.876471 | -0.002467 |
| `gulftoday_2026-08-21_page_0013.png:label_10` | 0.510766 | 0.505430 | -0.005336 |
| `gulftoday_2026-09-02_page_0012.png:label_3` | 0.801506 | 0.797939 | -0.003568 |

The three held-out July Gulf recovery targets also lost confidence:

| Recovery target | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `srtip_left_liquidation` | 0.434169 | 0.429048 | -0.005121 |
| `srtip_right_liquidation` | 0.718254 | 0.713411 | -0.004843 |
| `hamriyah_termination_shareholders` | 0.792699 | 0.789122 | -0.003577 |

## Diagnosis

The classification-only scope and frozen geometry avoided the catastrophic box
regression seen in V14, but the epoch-level gradient balance still favored
rejection. The recovery branch was present and active, yet its four unique
training references were too sparse to offset the shared confidence reduction
from the rejection objective. Continuing this trajectory would be expected to
improve rejection by lowering scores broadly rather than by separating reviewed
valid and excluded regions.

This is a model/data objective limitation, not an inference-geometry limitation:
the frozen v6 detector has feasible boxes for all 29 audited references,
including all three Gulf recovery targets.

## Run History

- R1 and R2 stopped before optimizer updates while real rectangular-batch
  `ratio_pad` metadata handling was corrected.
- R3 completed optimizer updates but was terminated by the original oversized
  checkpoint serialization path; no checkpoint was written.
- R4 completed, saved a criterion-free checkpoint, reloaded it successfully,
  ran all fixed-threshold suites, and stopped automatically without promotion.

## Closure

The V15 implementation and bounded pilot are complete. Any subsequent training
version requires a new plan that changes the recovery/rejection balance or adds
representative reviewed recovery supervision; increasing V15's epoch count is
not authorized by these results.
