"""Regression checks for the inactive-head bug and identity-based preservation."""
import unittest
import torch
from active_head_pilot import ActiveHeadLoss, stop_reasons


class ActiveHeadRegressionTests(unittest.TestCase):
    def test_only_operational_branch_receives_loss_gradients(self):
        # Exercise the adapter's branch selection, including eval tuple output.
        class Native:
            def parse_output(self, p):
                return p[1] if isinstance(p, tuple) else p

            def __call__(self, p, batch):
                return p["scores"].square().sum(), {}

        criterion = ActiveHeadLoss.__new__(ActiveHeadLoss)
        criterion.native = Native()
        for wrap in (False, True):
            active = torch.tensor([2.0], requires_grad=True)
            inactive = torch.tensor([3.0], requires_grad=True)
            raw = {"one2many": {"scores": active}, "one2one": {"scores": inactive}}
            loss, _ = criterion((None, raw) if wrap else raw, {})
            loss.backward()
            self.assertEqual(active.grad.item(), 4.0)
            self.assertIsNone(inactive.grad)

    def reports(self):
        fixed = {"totals": {"valid_missed": 0, "unreviewed_accepted": 0, "critical_boundary_failures": 0, "excluded_accepted": 17},
                 "gate_results": {"prediction_provenance_verified": True}}
        challenge = {"totals": {"valid_missed": 0, "unreviewed_accepted": 0, "excluded_accepted_regions": 7}}
        baseline = {"pages": [{"valid_matches": [{"reference_id": "previously_valid"}]}]}
        july = {"totals": {"valid_missed": 3, "unreviewed_accepted": 0},
                "pages": [{"valid_matches": [{"reference_id": "previously_valid"}]}]}
        return fixed, challenge, july, baseline

    def test_unrecovered_targets_do_not_trigger_first_epoch_futility(self):
        self.assertEqual(stop_reasons(*self.reports()), [])

    def test_recovered_target_cannot_offset_losing_a_previous_valid(self):
        f, c, j, b = self.reports()
        j["pages"][0]["valid_matches"] = [{"reference_id": "newly_recovered"}]
        self.assertIn("july_previously_valid_lost", stop_reasons(f, c, j, b))

    def test_unverified_preservation_results_cannot_continue(self):
        f, c, j, b = self.reports()
        f["gate_results"]["prediction_provenance_verified"] = False
        self.assertIn("fixed_provenance", stop_reasons(f, c, j, b))


if __name__ == "__main__":
    unittest.main()
