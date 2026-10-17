# Title: test_main.py
# Description: Unit tests for main.py.
# Run with: python -m unittest test_main -v
#
# Tests use a small synthetic DataFrame so they run quickly and never touch the real data/ folder.

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import main


def make_df(n=200, seed=0):
    """Census-style predictor rows where DIS_PHYSICAL depends on age, so models have something to learn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "PRTAGE": rng.integers(65, 86, n),
        "PESEX": rng.integers(1, 3, n),
        "PTDTRACE": rng.integers(1, 5, n),
        "PEMARITL": rng.integers(1, 7, n),
        "HEFAMINC": rng.integers(1, 17, n),
        "PEEDUCA": rng.integers(31, 47, n),
        "HETENURE": rng.integers(1, 4, n),
        "GESTFIPS": rng.choice([1, 6, 36, 48], n),
        "PEMLR": rng.integers(1, 8, n),
    })
    for target in main.TARGETS:
        df[target] = rng.integers(0, 2, n)
    df["DIS_PHYSICAL"] = (df["PRTAGE"] + rng.normal(0, 3, n) > 75).astype(int)
    return df


class TestTrainModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with mock.patch("builtins.print"):
            cls.models, cls.x_train, cls.x_test, cls.y_train, cls.y_test = main.train_models(make_df(), cv=2)

    def test_returns_four_fitted_models(self):
        self.assertEqual(set(self.models),
                         {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"})
        for model in self.models.values():
            self.assertEqual(len(model.predict(self.x_test)), len(self.x_test))

    def test_split_is_70_30_and_stratified(self):
        self.assertEqual(len(self.x_train), 140)
        self.assertEqual(len(self.x_test), 60)
        self.assertAlmostEqual(self.y_train.mean(), self.y_test.mean(), delta=0.02)

    def test_uses_only_predictor_features(self):
        self.assertEqual(list(self.x_train.columns), main.FEATURES)

    def test_handles_unseen_category_in_test_data(self):
        x = self.x_test.copy()
        x["GESTFIPS"] = 99  # state code never seen during training
        for model in self.models.values():
            model.predict(x)


class TestTestModels(unittest.TestCase):

    def setUp(self):
        with mock.patch("builtins.print"):
            self.models, _, self.x_test, _, self.y_test = main.train_models(make_df(), cv=2)
            self.results, self.best = main.test_models(self.models, self.x_test, self.y_test)

    def test_reports_expected_metrics_for_each_model(self):
        self.assertEqual(len(self.results), 4)
        for r in self.results:
            for key in ["Accuracy", "Balanced Accuracy", "Macro F1", "F1 Score",
                        "Recall", "Precision", "ROC AUC", "Confusion Matrix"]:
                self.assertIn(key, r)
            self.assertEqual(r["Confusion Matrix"].sum(), len(self.y_test))

    def test_best_model_has_highest_macro_f1(self):
        self.assertEqual(self.best["Macro F1"], max(r["Macro F1"] for r in self.results))

    def test_models_beat_chance_on_learnable_target(self):
        self.assertGreater(self.best["ROC AUC"], 0.7)


class TestMain(unittest.TestCase):

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("data")

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_skips_data_prep_when_predictor_file_exists(self):
        make_df().to_csv(main.PREDICTOR_FILE, index=False)
        real_train = main.train_models
        fast_train = lambda target: real_train(target=target, cv=2)

        with mock.patch("builtins.print"), \
                mock.patch.object(main.dataPrep, "main") as prep, \
                mock.patch.object(main, "train_models", side_effect=fast_train):
            summary = main.main()

        prep.assert_not_called()
        self.assertEqual(summary["Target"].tolist(), main.TARGETS)

    def test_stops_when_no_data_available(self):
        with mock.patch("builtins.print"), \
                mock.patch.object(main.dataPrep, "main", return_value=None) as prep:
            self.assertIsNone(main.main())
        prep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
