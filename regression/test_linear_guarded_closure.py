"""Tests for complete-page linear closure guards."""
import json
from pathlib import Path
import unittest

import torch

from linear_guarded_closure import all_review_pages, background_upper, constraint_violations, logit


ROOT = Path(__file__).resolve().parents[1]


class LinearGuardedClosureTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "regression/legal_notice_v13_linear_guarded_config.json").read_text(encoding="utf-8"))

    def test_pages_without_references_are_background_guarded(self):
        pages = all_review_pages(self.config)
        self.assertEqual(len(pages), 344)
        self.assertTrue(any(not page["records"] for page in pages))

    def test_low_teacher_background_anchor_cannot_cross_deployment(self):
        teacher = torch.tensor([logit(0.0001), logit(0.79), logit(0.90)])
        upper = background_upper(teacher, self.config)
        self.assertAlmostEqual(float(upper[0].sigmoid()), 0.75, places=6)
        self.assertAlmostEqual(float(upper[1].sigmoid()), 0.75, places=6)
        self.assertAlmostEqual(float(upper[2].sigmoid()), 0.903, places=6)
        violations = constraint_violations(
            torch.tensor([logit(0.80), logit(0.76), logit(0.904)]),
            torch.full((3,), -torch.inf),
            upper,
        )
        self.assertTrue((violations > 0).all())

    def test_valid_anchor_has_lower_and_upper_bound(self):
        lower = torch.tensor([logit(0.805)])
        upper = torch.tensor([logit(0.90)])
        self.assertGreater(float(constraint_violations(torch.tensor([logit(0.79)]), lower, upper)[0]), 0)
        self.assertEqual(float(constraint_violations(torch.tensor([logit(0.85)]), lower, upper)[0]), 0)
        self.assertGreater(float(constraint_violations(torch.tensor([40.0]), lower, upper)[0]), 0)

    def test_each_anchor_is_checked_independently(self):
        lower = torch.full((2,), -torch.inf)
        upper = torch.tensor([logit(0.49), logit(0.49)])
        violations = constraint_violations(torch.tensor([logit(0.20), logit(0.90)]), lower, upper)
        self.assertEqual(float(violations[0]), 0)
        self.assertGreater(float(violations[1]), 0)


if __name__ == "__main__":
    unittest.main()
