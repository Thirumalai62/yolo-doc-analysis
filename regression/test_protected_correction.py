"""Check corrective score direction, preservation, and overlay loading."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from protected_correction import objective, nonlinear_logits, overlay_train_pages


class ProtectedCorrectionTests(unittest.TestCase):
    def gradient(self, kind, teacher, change=0.0):
        config = {"target_table_probability": 0.97, "target_negative_probability": 0.5,
                  "positive_logit_tolerance": 0.05, "background_logit_tolerance": 0.15,
                  "guard_weight": 25.0, "ridge_weight": 0.01}
        delta = torch.zeros(3, 257, requires_grad=True)
        with torch.no_grad():
            delta[0, 0] = change
        x = torch.zeros(1, 257)
        x[0, 0] = 1.0
        bucket = {"x": x, "teacher": torch.tensor([teacher]), "kind": torch.tensor([kind]),
                  "group": torch.tensor([0 if kind >= 2 else -1])}
        loss, _ = objective(delta, [bucket], config)
        loss.backward()
        return delta.grad[0, 0].item()

    def test_missed_positive_is_pushed_up(self):
        self.assertLess(self.gradient(2, 0.0), 0)

    def test_procurement_is_pushed_down(self):
        self.assertGreater(self.gradient(3, 3.0), 0)

    def test_valid_score_drop_is_opposed(self):
        self.assertLess(self.gradient(1, 3.0, -0.5), 0)

    def test_background_score_increase_is_opposed(self):
        self.assertGreater(self.gradient(0, -3.0, 0.5), 0)

    def test_unchanged_working_detection_has_no_gradient(self):
        self.assertEqual(self.gradient(1, 3.0), 0)

    def test_low_score_negatives_cannot_dilute_high_score_failure(self):
        config = {"target_negative_probability": 0.5, "guard_weight": 0.0,
                  "ridge_weight": 0.0, "negative_pooling": "max"}
        losses = []
        for low_count in (0, 100):
            n = low_count + 1
            bucket = {"x": torch.zeros(n, 257), "teacher": torch.tensor([3.0] + [-6.0] * low_count),
                      "kind": torch.full((n,), 3), "group": torch.zeros(n, dtype=torch.long)}
            loss, _ = objective(torch.zeros(3, 257), [bucket], config)
            losses.append(loss.item())
        self.assertEqual(losses, [9.0, 9.0])

    def test_cached_nonlinear_block_matches_native_forward_and_gradients(self):
        from ultralytics.nn.modules.conv import Conv
        torch.manual_seed(0)
        block = Conv(4, 4, 1).eval()
        final = torch.nn.Conv2d(4, 1, 1)
        x = torch.randn(11, 4)
        reference = final(block(x[:, :, None, None])).flatten()
        actual = nonlinear_logits(x, block, final)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
        parameters = list(block.parameters()) + list(final.parameters())
        native_gradients = torch.autograd.grad(reference.sum(), parameters)
        cached_gradients = torch.autograd.grad(actual.sum(), parameters)
        for a, b in zip(native_gradients, cached_gradients):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_hashed_overlay_supplies_labeled_and_empty_training_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "overlay.json"
            manifest = {
                "status": "prepared_not_trained",
                "class_name": "legal_notice",
                "pages": [
                    {"issue": "paper_2026-01-01", "image_name": "paper_2026-01-01_page_0001.png",
                     "image": "positive.png", "label": "positive.txt", "valid_notice_count": 1,
                     "image_sha256": "image", "label_sha256": "label"},
                    {"issue": "paper_2026-01-02", "image_name": "paper_2026-01-02_page_0001.png",
                     "image": "negative.png", "label": "negative.txt", "valid_notice_count": 0,
                     "image_sha256": "image", "label_sha256": "label"},
                ],
            }
            content = json.dumps(manifest).encode()
            manifest_path.write_bytes(content)
            config = {
                "correction_overlay_manifest": "overlay.json",
                "correction_overlay_manifest_sha256": hashlib.sha256(content).hexdigest(),
            }
            with patch("protected_correction.ROOT", root):
                pages: list[dict] = overlay_train_pages(config)
            self.assertTrue(pages[0]["reviewed_complete_labels"])
            self.assertFalse(pages[0]["reviewed_no_valid_notices"])
            self.assertFalse(pages[1]["reviewed_complete_labels"])
            self.assertTrue(pages[1]["reviewed_no_valid_notices"])


if __name__ == "__main__":
    unittest.main()
