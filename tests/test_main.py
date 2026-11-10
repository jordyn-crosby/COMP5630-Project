# Title: test_main.py
# Description: Unit tests for main.py.
# Run with: python -m unittest tests.test_main -v
#
# Tests use a small synthetic DataFrame so they run quickly and never touch the real data/ folder.

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import main


def make_df(n=200, seed=0):
    """Census-style predictor rows where DIG_EXCLUDED depends on age and income, so models have something to learn."""
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
    for col in main.DISABILITY_FEATURES:
        df[col] = rng.integers(0, 2, n)
    df["HRYEAR4"] = rng.choice([2019, 2021, 2023], n)
    df[main.TARGET] = (df["PRTAGE"] - df["HEFAMINC"] + rng.normal(0, 3, n) > 67).astype(int)
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

    def test_features_are_the_ten_predictors_without_internet_columns(self):
        self.assertEqual(len(main.NUMERIC_FEATURES + main.CATEGORICAL_FEATURES) + 1, 10)
        for col in main.FEATURES:
            self.assertFalse(col.startswith(("PEIN", "HEIN", "PENET", "HENET")), col)
        self.assertNotIn(main.TARGET, main.FEATURES)
        self.assertNotIn(main.YEAR_COLUMN, main.FEATURES)

    def test_handles_unseen_category_in_test_data(self):
        x = self.x_test.copy()
        x["GESTFIPS"] = 99  # state code never seen during training
        for model in self.models.values():
            model.predict(x)


class TestTemporalSplit(unittest.TestCase):

    def test_trains_on_earlier_years_and_tests_on_latest(self):
        df = make_df()
        with mock.patch("builtins.print"):
            models, x_train, x_test, _, _ = main.train_models_temporal(df, cv=2)
        years = df[main.YEAR_COLUMN]
        self.assertTrue(years.loc[x_train.index].isin(main.TEMPORAL_TRAIN_YEARS).all())
        self.assertTrue((years.loc[x_test.index] == main.TEMPORAL_TEST_YEAR).all())
        self.assertEqual(len(x_train) + len(x_test), len(df))
        self.assertEqual(len(models), 4)


class TestFeatureImportance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with mock.patch("builtins.print"):
            cls.models = main.train_models(make_df(), cv=2)[0]
        cls.importance = main.feature_importance(cls.models)

    def test_one_row_per_predictor(self):
        expected = set(main.NUMERIC_FEATURES + main.CATEGORICAL_FEATURES) | {main.DISABILITY_GROUP}
        self.assertEqual(set(self.importance.index), expected)

    def test_each_model_normalized_and_sorted(self):
        for name in self.models:
            self.assertAlmostEqual(self.importance[name].sum(), 1.0)
        self.assertTrue(self.importance["Mean"].is_monotonic_decreasing)

    def test_learnable_predictors_rank_highest(self):
        self.assertEqual(set(self.importance.index[:2]), {"PRTAGE", "HEFAMINC"})

    def test_source_feature_mapping(self):
        self.assertEqual(main.source_feature("num__PRTAGE"), "PRTAGE")
        self.assertEqual(main.source_feature("cat__GESTFIPS_36"), "GESTFIPS")
        self.assertEqual(main.source_feature("dis__DIS_VISION"), main.DISABILITY_GROUP)

    def test_expand_features(self):
        self.assertEqual(main.expand_features(["PRTAGE", main.DISABILITY_GROUP]),
                         ["PRTAGE"] + main.DISABILITY_FEATURES)

    def test_reduced_model_trains_on_subset(self):
        features = main.expand_features(["PRTAGE", "GESTFIPS", main.DISABILITY_GROUP])
        with mock.patch("builtins.print"):
            models, x_train, x_test, _, _ = main.train_models(make_df(), cv=2, features=features)
        self.assertEqual(list(x_train.columns), features)
        for model in models.values():
            model.predict(x_test)


class TestTestModels(unittest.TestCase):

    def setUp(self):
        with mock.patch("builtins.print"):
            self.models, _, self.x_test, _, self.y_test = main.train_models(make_df(), cv=2)
            self.results, self.best = main.test_models(self.models, self.x_test, self.y_test)

    def test_reports_expected_metrics_for_each_model(self):
        self.assertEqual(len(self.results), 5)  # four models plus the baseline
        self.assertEqual(self.results[-1]["Model"], main.BASELINE_NAME)
        for r in self.results:
            for key in ["Accuracy", "Balanced Accuracy", "Macro F1", "F1 Score",
                        "Recall", "Precision", "ROC AUC", "Confusion Matrix"]:
                self.assertIn(key, r)
            self.assertEqual(r["Confusion Matrix"].sum(), len(self.y_test))

    def test_best_model_is_never_the_baseline(self):
        self.assertNotEqual(self.best["Model"], main.BASELINE_NAME)

    def test_results_table_counts_match_accuracy(self):
        table = main.results_table(self.results, "Random split")
        self.assertEqual(len(table), 5)
        counts = table[["Correct Connected (TN)", "False Alarms (FP)",
                        "Missed Excluded (FN)", "Correct Excluded (TP)"]].sum(axis=1)
        self.assertTrue((counts == len(self.y_test)).all())
        self.assertTrue((table["Test Size"] == len(self.y_test)).all())
        correct = table["Correct Connected (TN)"] + table["Correct Excluded (TP)"]
        for acc, c in zip(table["Accuracy"], correct / len(self.y_test)):
            self.assertAlmostEqual(acc, c, places=3)
        self.assertNotIn("Classification Report", table.columns)

    def test_best_model_has_highest_macro_f1(self):
        self.assertEqual(self.best["Macro F1"], max(r["Macro F1"] for r in self.results))

    def test_models_beat_chance_on_learnable_target(self):
        self.assertGreater(self.best["ROC AUC"], 0.7)


class TestFormatConfusionMatrix(unittest.TestCase):

    def test_labels_counts_and_accuracy(self):
        text = main.format_confusion_matrix(np.array([[50, 10], [5, 35]]))
        lines = text.splitlines()
        self.assertIn("n=100", lines[0])
        self.assertIn("Actually connected", lines[2])
        self.assertIn("50 (correct)", lines[2])
        self.assertIn("10 (false alarm)", lines[2])
        self.assertIn("5 (missed)", lines[3])
        self.assertIn("35 (correct)", lines[3])
        self.assertEqual(lines[4], "Overall accuracy: 85 / 100 = 0.850")


class TestMain(unittest.TestCase):

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("data")
        tmp = Path(self._tmp.name)
        for name, path in [("PREDICTOR_FILE", tmp / "data" / "predictor_data.csv"),
                           ("RESULTS_FILE", tmp / "results" / "model_performance.csv")]:
            patcher = mock.patch.object(main, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_skips_data_prep_when_predictor_file_exists(self):
        make_df().to_csv(main.PREDICTOR_FILE, index=False)
        real_fit = main.fit_models
        fast_fit = lambda x, y, cv=5: real_fit(x, y, cv=2)

        with mock.patch("builtins.print"), \
                mock.patch.object(main.dataPrep, "main") as prep, \
                mock.patch.object(main, "fit_models", side_effect=fast_fit):
            summary = main.main()

        prep.assert_not_called()
        self.assertEqual(summary["Experiment"].tolist(),
                         [main.RANDOM_EXPERIMENT, main.TEMPORAL_EXPERIMENT, main.REDUCED_EXPERIMENT])

        table = pd.read_csv(main.RESULTS_FILE)
        self.assertEqual(len(table), 15)  # 3 experiments x (4 models + baseline)
        self.assertTrue(os.path.getsize("results/model_performan
