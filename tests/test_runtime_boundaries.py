import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_package_does_not_import_main(self):
        for path in (ROOT / "doc_detector").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertNotIn("main", {alias.name for alias in node.names}, path)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "main", path)

    def test_runtime_image_context_excludes_main(self):
        dockerignore = (ROOT / "Dockerfile.opensandbox.dockerignore").read_text(
            encoding="utf-8"
        )
        self.assertEqual(dockerignore.splitlines()[0], "**")
        self.assertNotIn("!main.py", dockerignore.splitlines())


if __name__ == "__main__":
    unittest.main()
