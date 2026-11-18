"""Small software fixtures only; never use reserved evaluator trips."""
import unittest
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from pathsnap_real_training import FEATURES, K, SEED, models


class TrainingContractTests(unittest.TestCase):
    def test_feature_schema(self):
        self.assertEqual(K, 5)
        self.assertEqual(FEATURES, [
            'distance_m', 'bearing_difference_deg', 'speed_mps',
            'arc_length_m', 'projection_fraction', 'time_gap_s',
            'candidate_rank',
        ])
        self.assertEqual(SEED, 42)

    def test_grouped_validation_is_disjoint(self):
        group = np.repeat(np.arange(10), 4)
        features = np.zeros((len(group), K, len(FEATURES)))
        validation_indices = []
        for tr, va in GroupKFold(n_splits=5).split(features, groups=group):
            self.assertFalse(set(group[tr]) & set(group[va]))
            validation_indices.extend(va.tolist())
        self.assertEqual(sorted(validation_indices), list(range(len(group))))

    def test_five_estimators_fit_and_output_binary_probabilities(self):
        rng = np.random.default_rng(SEED)
        x = rng.normal(size=(100, K, len(FEATURES)))
        y = np.zeros((100, K), dtype=int)
        y[np.arange(100), np.arange(100) % K] = 1
        x[:, :, 0] -= y
        estimators = models()
        self.assertEqual(set(estimators), {
            'Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree',
            'Random Forest', 'Histogram Gradient Boosting',
        })
        for name, estimator in estimators.items():
            with self.subTest(model=name):
                fitted = clone(estimator).fit(x[:80].reshape(-1, len(FEATURES)), y[:80].ravel())
                p = fitted.predict_proba(x[80:].reshape(-1, len(FEATURES)))
                np.testing.assert_array_equal(fitted.classes_, [0, 1])
                self.assertEqual(p.shape, (20 * K, 2))
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                np.testing.assert_allclose(p.sum(axis=1), 1)
                self.assertEqual(p[:, 1].reshape(-1, K).argmax(axis=1).shape, (20,))


if __name__ == '__main__':
    unittest.main()
