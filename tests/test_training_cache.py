import unittest
import numpy as np
from training_cache import validate_development


class CacheValidationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((80, 5, 7))
        self.y = np.zeros((80, 5), dtype=int)
        self.y[:, 0] = 1
        self.g = np.arange(80)
        self.tx = np.zeros((20, 5, 7))
        self.ty = np.zeros((20, 5), dtype=int)
        self.ty[:, 0] = 1
        self.tg = np.repeat(np.arange(80, 90), 2)
        self.ids = np.zeros((20, 5), dtype=int)

    def validate(self):
        validate_development(self.x, self.y, self.g, self.tx, self.ty,
                             self.tg, self.ids, self.ids, self.ids)

    def test_valid_cache_and_candidate_miss(self):
        self.y[0] = 0
        self.validate()

    def test_reject_multiple_positives(self):
        self.y[0, 1] = 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_split_overlap(self):
        self.tg[:2] = 0
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            self.validate()

    def test_reject_nan(self):
        self.x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_wrong_candidate_shape(self):
        self.ty = self.ty[:, :4]
        with self.assertRaises(ValueError):
            self.validate()
