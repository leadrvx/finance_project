import unittest

from fraud_detector import (
    create_visualizations,
    generate_sample_data,
    plot_confusion_matrix,
    prepare_features,
    run_eda,
    train_model,
)


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

        self.assertIn("roc_auc", artifacts.metrics)

    def test_eda_visualizations_and_confusion_matrix(self):
        data = generate_sample_data(n_samples=300, random_state=4)

        eda = run_eda(data)
        for key in ("shape", "missing_values", "class_distribution", "numeric_summary"):
            self.assertIn(key, eda)

        figures = create_visualizations(data)
        self.assertGreaterEqual(len(figures), 1)
        self.assertTrue(all(hasattr(fig, "axes") for fig in figures))

        artifacts = train_model(data)
        cm_fig = plot_confusion_matrix(artifacts.y_test, artifacts.y_pred)
        self.assertTrue(hasattr(cm_fig, "axes"))


if __name__ == "__main__":
    unittest.main()
