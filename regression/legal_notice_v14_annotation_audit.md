# V14 Annotation Coverage Audit

## Result

The consolidated epoch-training dataset contains 149 unique reviewed full pages,
149 labels, and 753 legal-notice boxes. Its 91 train pages contain 324 boxes;
the 28 validation pages contain 162 boxes; the locked 30-page historical test
split contains 267 boxes.

All seven September 1-8 corrective pages are present in train. The two mixed
vehicle-auction pages retain all 21 reviewed legal notices, while the five
fully negative news/header/tender pages have explicit empty labels.

## Canonical Sources

| Source | Reviewed pages represented | Notes |
|---|---:|---|
| `initial_cvat_review` | 6/6 | One label was later corrected. |
| `legal_notice_v2_review` | 50/50 | Three labels were later corrected. |
| `legal_notice_v3_corrected_review` | 47/48 | Corrected export supersedes `legal_notice_v3_new_sources_review`. |
| `legal_notice_v4_robustness` | 36/37 | One duplicate vehicle-disposal page was intentionally omitted. |
| `legal_notice_v10_targeted_correction` | 7/7 | Five negatives and two mixed positive pages. |
| Reviewed non-CVAT procurement extras | 3/3 | Fully reviewed empty-label pages. |

The two omitted reviewed pages are documented upstream:

- `khaleejtimes_2026-09-04_page_0014.png` repeats a notice already assigned to
  train; omission prevents train/test leakage.
- `gulftoday_2026-08-21_page_0006.png` duplicates the same Ajman vehicle
  publication represented by the reviewed Al Bayan pages.

## Edge-Case Coverage

| Family | Represented coverage |
|---|---:|
| News/editorial/header negatives | 43 empty-label pages |
| Vehicle auction/disposal | 8 pages, including 3 mixed positive pages |
| Tender/e-tender/procurement | 16 pages, including 1 mixed positive page |
| Explicit mixed pages | 17 pages with 99 preserved legal-notice boxes |

The effective train schedule has 109 samples from 91 physical pages. It repeats
13 reviewed hard-negative or preservation pages but does not duplicate files.

## Training Decision

No image or label correction is required before the V14 run. V14 uses
`dataset_v11_consolidated` unchanged and pins its hashes. September correction
issues are development data, not independent holdout evidence. The historical
test split remains locked until checkpoint selection.

The V14 trainer starts from v6. V8 and v10 are retained only as regression
evidence: v8 demonstrates the Gulf Today table recovery, while v10 demonstrates
that narrow score correction does not generalize to the September 9 cases.
