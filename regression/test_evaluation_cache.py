"""Lifecycle checks for issue-scoped evaluation caching."""
import unittest
from unittest.mock import call, patch

import evaluation_cache


class IssueScopedCacheTests(unittest.TestCase):
    def test_switching_issue_cleans_previous_cache(self):
        root = evaluation_cache.ROOT / "output/regression_audit/cache_test"
        cache = evaluation_cache.IssueScopedCache({}, "fixed_manifest", root)
        with patch("evaluation_cache.prepare") as prepare, patch("evaluation_cache.cleanup") as cleanup:
            cache.activate({"issue": "issue_a"})
            cache.activate({"issue": "issue_a"})
            cache.activate({"issue": "issue_b"})
            cache.close()

        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(
            cleanup.call_args_list,
            [call(root / "fixed_manifest_issue_a.json"), call(root / "fixed_manifest_issue_b.json")],
        )


if __name__ == "__main__":
    unittest.main()
