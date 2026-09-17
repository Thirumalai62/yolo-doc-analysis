"""Check that the V17 evaluator covers the complete reviewed release contract."""
from __future__ import annotations

import unittest
from pathlib import Path

from evaluate_v17_release import greedy_matches, load, project_path, unresolved_additions


class V17ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load(project_path("regression/legal_notice_v17_release_config.json"))

    def manifest(self, name):
        return load(project_path(self.config["manifests"][name]["path"]))

    def test_static_inventory_covers_every_required_family(self):
        requirements = self.config["requirements"]
        fixed = self.manifest("fixed")
        challenge = self.manifest("challenge")
        july = self.manifest("july")
        target = self.manifest("target")
        historical = self.manifest("historical")
        semantic = self.manifest("semantic_reviews")
        september11 = self.manifest("september11")
        self.assertEqual(sum(reference["classification"] == "valid" for page in fixed["pages"] for reference in page["references"]), requirements["fixed_valid"])
        self.assertEqual(sum(reference["classification"] == "excluded" for page in fixed["pages"] for reference in page["references"]), 17)
        self.assertEqual(sum(len(page["valid"]) for page in challenge["pages"]), requirements["challenge_valid"])
        self.assertEqual(sum(len(page["excluded"]) for page in challenge["pages"]), 13)
        self.assertEqual(sum(len(page["valid"]) for page in july["pages"]), requirements["july_valid"])
        self.assertEqual(sum(len(page["excluded"]) for page in july["pages"]), 2)
        self.assertEqual(sum(page["valid_notice_count"] for page in target["pages"]), requirements["target_valid"])
        self.assertEqual(sum(len(page["negative_regions_xyxyn"]) for page in target["pages"]), requirements["target_negative_regions"])
        self.assertEqual(sum(reference["classification"] == "valid" for page in historical["pages"] for reference in page["references"]), requirements["historical_valid"])
        self.assertEqual(sum(reference["classification"] == "excluded" for page in historical["pages"] for reference in page["references"]), requirements["historical_excluded"])
        self.assertEqual(sum(review["classification"] == "valid" for review in semantic["reviews"]), requirements["semantic_valid"])
        self.assertEqual(sum(review["classification"] == "excluded" for review in semantic["reviews"]), requirements["semantic_excluded"])
        self.assertEqual(len(september11["regions"]), requirements["september11_excluded"])

    def test_september15_labels_supply_all_nineteen_references(self):
        manifest = self.manifest("september15")
        count = 0
        for page in manifest["pages"]:
            count += len([line for line in project_path(page["labels"]).read_text(encoding="utf-8").splitlines() if line.strip()])
        self.assertEqual(count, self.config["requirements"]["september15_valid"])

    def test_critical_recoveries_use_the_deployment_acceptance_contract(self):
        source = project_path("regression/evaluate_v17_release.py").read_text(encoding="utf-8")
        self.assertNotIn("0.9 if critical", source)
        self.assertNotIn("0.85 if critical", source)

    def test_greedy_matching_cannot_reuse_prediction(self):
        references = [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [1, 1, 11, 11]}]
        predictions = [{"xyxy": [0, 0, 10, 10], "confidence": 0.9}]
        matches = greedy_matches(references, predictions, 0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len({match[0] for match in matches.values()}), 1)

    def test_reviewed_addition_resolves_matching_unreviewed_detection(self):
        report = {"pages": [{"issue": "paper_2026-01-01", "page": 1, "unreviewed_accepted": [{"xyxy": [0, 0, 10, 10], "confidence": 0.9}]}]}
        reviews = {"reviews": [{"suite": "fixed080", "issue": "paper_2026-01-01", "page": 1, "xyxy": [0, 0, 10, 10], "classification": "valid"}]}
        self.assertEqual(unresolved_additions(report, reviews, "fixed080"), [])


if __name__ == "__main__":
    unittest.main()
