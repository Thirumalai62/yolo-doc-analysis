"""Constraint semantics for the bounded closure correction."""
import unittest

import torch

from constrained_closure_correction import constraint_state


class ClosureConstraintTests(unittest.TestCase):
    def test_each_negative_region_has_an_independent_ceiling(self):
        cache = {"records": [
            {"kind": "negative", "required_probability": 0.49},
            {"kind": "negative", "required_probability": 0.49},
        ]}
        scores = torch.logit(torch.tensor([0.20, 0.90]))
        violations, valid, _ = constraint_state(scores, cache)
        self.assertEqual(valid.tolist(), [False, False])
        self.assertEqual(float(violations[0]), 0.0)
        self.assertGreater(float(violations[1]), 0.0)

    def test_one_failing_valid_reference_cannot_be_hidden(self):
        cache = {"records": [
            {"kind": "valid", "required_probability": 0.805},
            {"kind": "valid", "required_probability": 0.805},
        ]}
        scores = torch.logit(torch.tensor([0.99, 0.79]))
        violations, valid, _ = constraint_state(scores, cache)
        self.assertTrue(valid.all())
        self.assertEqual(float(violations[0]), 0.0)
        self.assertGreater(float(violations[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
