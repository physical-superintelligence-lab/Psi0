import tempfile
import unittest
from pathlib import Path

from scripts.offline.sonic_schema import detect_end_effector


class SonicEndEffectorDetectionTest(unittest.TestCase):
    def test_auto_defaults_to_dex3_without_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            self.assertEqual(detect_end_effector(root, {"action": {}}), "dex3")

    def test_auto_detects_dex1_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            modality = {"end_effector": "dex1_1", "action": {}}
            self.assertEqual(detect_end_effector(root, modality), "dex1_1")


if __name__ == "__main__":
    unittest.main()
