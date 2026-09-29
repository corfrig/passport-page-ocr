import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.passport_page_detector import find_page


class DetectPageSmokeTests(unittest.TestCase):
    def check(self, relative_path):
        image = cv2.imread(str(ROOT / relative_path))
        self.assertIsNotNone(image)
        return find_page(image)

    def test_page_geometry_from_each_group(self):
        good = self.check("data/opensource_selection/good/001.png")
        normal = self.check("data/opensource_selection/normal/001.png")
        bad = self.check("data/opensource_selection/bad/021.png")
        self.assertTrue(all(item["found"] for item in (good, normal, bad)))
        self.assertEqual(good["area_fraction"], 1.0)
        self.assertGreater(normal["area_fraction"], 0.10)
        self.assertGreater(bad["area_fraction"], 0.10)
        self.assertTrue(all(len(item["corners"]) == 4 for item in (good, normal, bad)))

    def test_document_free_photos_are_rejected(self):
        self.assertFalse(self.check("tests/negative/cat.jpg")["found"])
        self.assertFalse(self.check("tests/negative/mountains.jpg")["found"])


if __name__ == "__main__":
    unittest.main()
