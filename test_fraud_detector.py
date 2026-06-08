import unittest

from fraud_detector import generate_sample_data, prepare_features, train_model


class FraudDetectorTests(unittest.TestCase):
    def test_generate_sample_data_has_target(self):
        data = generate_sample_data(n_samples=100, random_state=1)
        self.assertIn("is_fraud", data.columns)
        self.assertTrue(set(data["is_fraud"].unique()).issubset({0, 1}))

    def test_prepare_features_excludes_target(self):
        data = generate_sample_data(n_samples=200, random_state=2)
        X, y = prepare_features(data)
        self.assertNotIn("is_fraud", X.columns)
        self.assertEqual(len(X), len(y))

    def test_train_model_returns_metrics(self):
        data = generate_sample_data(n_samples=600, random_state=3)
        artifacts = train_model(data)
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(name, artifacts.metrics)
            self.assertGreaterEqual(artifacts.metrics[name], 0.0)
            self.assertLessEqual(artifacts.metrics[name], 1.0)


if __name__ == "__main__":
    unittest.main()
