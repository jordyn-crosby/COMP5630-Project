# Title: test_main.py
# Description: Unit tests for main.py (splits, folds, feature exclusions, training, evaluation, outputs).
# Run with: python -m unittest test_main -v
#
# Tests use a small synthetic modeling table so they run quickly and never touch the real data/ folder.

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import main


def make_df(n=400, seed=0):
    """Synthetic predictor_data.csv rows: exclusion depends on age and education so models can learn."""
    rng = np.random.default_rng(seed)
    n_households = n // 2  # two cohort members per household on average
    hh = rng.integers(0, n_households, n)
    df = pd.DataFrame({
        "HRYEAR4": rng.choice([2019, 2021, 2023], n),
        "HRHHID": hh, "HRHHID2": 1, "PULINENO": rng.integers(1, 4, n), "HRNUMHOU": 2,
        "PWSSWGT": rng.uniform(500, 3000, n),
        "PRTAGE": rng.integers(65, 86, n),
        "PESEX": rng.integers(1, 3, n),
        "PTDTRACE": rng.integers(1, 5, n),
        "PEMARITL": rng.integers(1, 7, n),
        "HEFAMINC": rng.integers(1, 17, n),
        "PEEDUCA": rng.integers(31, 47, n),
        "HETENURE": rng.integers(1, 4, n),
        "GESTFIPS": rng.choice([1, 6, 13, 36, 48], n),
        "PEMLR": rng.integers(1, 8, n),
    })
    for col in main.BINARY_FEATURES:
        df[col] = rng.integers(0, 2, n)
    df["DISTYPE"] = "Mobility"
    score = (df["PRTAGE"] - 75) / 5 - (df["PEEDUCA"] - 39) / 4 + rng.normal(0, 0.8, n)
    df[main.TARGET] = (score > 0).astype(int)
    return df


class TestSplits(unittest.TestCase):

    def setUp(self):
        self.df = make_df()

    def test_pooled_split_is_about_70_30_stratified_and_household_disjoint(self):
        train, test = main.pooled_split(self.df)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertAlmostEqual(len(test) / len(self.df), 0.30, delta=0.04)
        self.assertAlmostEqual(train[main.TARGET].mean(), test[main.TARGET].mean(), delta=0.06)
        self.assertTrue(main.check_disjoint_groups(train, test))

    def test_pooled_split_is_reproducible(self):
        a_train, a_test = main.pooled_split(self.df)
        b_train, b_test = main.pooled_split(self.df)
        pd.testing.assert_frame_equal(a_train, b_train)
        pd.testing.assert_frame_equal(a_test, b_test)

    def test_temporal_split_uses_2019_2021_for_development_and_2023_for_test(self):
        dev, test = main.temporal_split(self.df)
        self.assertEqual(set(dev["HRYEAR4"]), {2019, 2021})
        self.assertEqual(set(test["HRYEAR4"]), {2023})
        self.assertEqual(len(dev) + len(test), len(self.df))

    def test_cv_folds_keep_households_together(self):
        train, _ = main.pooled_split(self.df)
        groups = main.make_group_key(train).to_numpy()
        for tr, va in main.cv_folds(train, k=3):
            self.assertFalse(set(groups[tr]) & set(groups[va]))

    def test_group_key_separates_years(self):
        df = pd.DataFrame({"HRYEAR4": [2019, 2023], "HRHHID": [5, 5], "HRHHID2": [1, 1]})
        self.assertEqual(main.make_group_key(df).nunique(), 2)

    def test_disjoint_check_fails_on_overlap(self):
        with self.assertRaises(AssertionError):
            main.check_disjoint_groups(self.df.iloc[:10], self.df.iloc[:10])


class TestFeatureDefinitions(unittest.TestCase):

    def test_fifteen_inputs_across_ten_groups(self):
        self.assertEqual(len(main.FEATURES), 15)
        self.assertEqual(len(main.BINARY_FEATURES), 6)
        self.assertEqual(len(set(main.FEATURE_GROUPS[f] for f in main.FEATURES)), 10)

    def test_no_internet_or_outcome_fields_in_inputs(self):
        self.assertFalse(set(main.FEATURES) & main.INTERNET_COLUMNS)
        self.assertNotIn(main.TARGET, main.FEATURES)
        for col in ["HRHHID", "HRHHID2", "HRYEAR4", "PWSSWGT", "PULINENO"]:
            self.assertNotIn(col, main.FEATURES)

    def test_reduced_set_is_education_income_age_disability(self):
        self.assertEqual(set(main.REDUCED_FEATURES), {"PRTAGE", "HEFAMINC", "PEEDUCA"} | set(main.BINARY_FEATURES))

    def test_load_predictor_table_rejects_leaked_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("data")
                df = make_df(50)
                df["PEINHOME"] = 1
                df.to_csv(main.PREDICTOR_FILE, index=False)
                with self.assertRaises(ValueError):
                    main.load_predictor_table()
            finally:
                os.chdir(cwd)


class TestTrainAndTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = make_df()
        cls.train, cls.test = main.pooled_split(cls.df)
        folds = main.cv_folds(cls.train, k=2)
        with mock.patch("builtins.print"):
            cls.searches, cls.dummy, cls.cv_summary, cls.cv_folds_table = main.train_models(
                cls.train, folds, quick=True, verbose=False)
            cls.results, cls.predictions = main.test_models(cls.searches, cls.dummy, cls.test, verbose=False)

    def test_returns_four_tuned_models_and_a_dummy(self):
        self.assertEqual(set(self.searches), {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"})
        self.assertEqual(self.dummy.predict(self.test[main.FEATURES]).tolist(),
                         [self.train[main.TARGET].mode()[0]] * len(self.test))

    def test_cv_summary_has_one_row_per_model_and_fold_table_per_fold(self):
        self.assertEqual(len(self.cv_summary), 4)
        self.assertEqual(len(self.cv_folds_table), 8)

    def test_results_include_dummy_and_promised_metrics(self):
        self.assertEqual([r["Model"] for r in self.results], main.MODEL_ORDER)
        for r in self.results:
            for key in ["Accuracy", "Precision (excluded)", "Recall (excluded)", "F1 (excluded)", "ROC AUC",
                        "Macro F1", "Balanced Accuracy", "TN", "FP", "FN", "TP"]:
                self.assertIn(key, r)
            self.assertEqual(r["TN"] + r["FP"] + r["FN"] + r["TP"], len(self.test))

    def test_predictions_cover_every_test_record_with_ids(self):
        self.assertEqual(len(self.predictions), len(self.test))
        for col in main.ID_COLUMNS + [main.TARGET, "pred_Dummy", "pred_XGBoost", "prob_Random_Forest"]:
            self.assertIn(col, self.predictions.columns)

    def test_models_beat_dummy_on_learnable_target(self):
        dummy_auc = self.results[0]["ROC AUC"]
        best = max(r["ROC AUC"] for r in self.results[1:])
        self.assertGreater(best, max(dummy_auc, 0.7))

    def test_handles_unseen_category_in_test_data(self):
        x = self.test[main.FEATURES].copy()
        x["GESTFIPS"] = 99
        for search in self.searches.values():
            search.best_estimator_.predict(x)

    def test_logistic_coefficients_table(self):
        table = main.logistic_coefficients(self.searches["Logistic Regression"], self.train)
        self.assertIn("odds_ratio", table.columns)
        self.assertTrue((table[table["input"] == "PRTAGE"]["coef_per_unit"].notna()).all())
        state = table[table["input"] == "GESTFIPS"]
        self.assertTrue(state["coef_vs_reference"].notna().all())
        self.assertAlmostEqual(state["coef_vs_reference"].abs().min(), 0.0)  # the reference level itself

    def test_tree_importances_aggregate_to_raw_inputs(self):
        table = main.tree_importances(self.searches)
        for model in ["Decision Tree", "Random Forest", "XGBoost"]:
            sub = table[table["Model"] == model]
            self.assertEqual(set(sub["input"]), set(main.FEATURES))
            self.assertAlmostEqual(sub["importance"].sum(), 1.0, places=3)

    def test_permutation_importance_one_row_per_raw_input(self):
        table = main.permutation_importances(self.searches, self.test, n_repeats=2)
        self.assertEqual(len(table), 4 * len(main.FEATURES))

    def test_hypotheses_table(self):
        table = main.evaluate_hypotheses(self.results)
        self.assertEqual((table["Hypothesis"] == "H1").sum(), 5)
        self.assertEqual((table["Hypothesis"] == "H2").sum(), 3)


class TestRunExperimentOutputs(unittest.TestCase):

    def test_saves_all_promised_artifacts(self):
        df = make_df()
        train, test = main.temporal_split(df)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            main.run_experiment("smoke", train, test, tmp, "smoke test", cv=2, quick=True, verbose=False)
            for name in ["model_comparison.csv", "test_predictions.csv", "best_params.json", "cv_summary.csv",
                         "cv_folds.csv", "confusion_matrices.csv", "classification_reports.txt", "hypotheses.csv",
                         "logistic_coefficients.csv", "tree_importances.csv", "permutation_importance.csv",
                         "run_manifest.json"]:
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            with open(os.path.join(tmp, "run_manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["internet_fields_in_inputs"], [])
            self.assertEqual(manifest["test"]["years"], {"2023": int((df["HRYEAR4"] == 2023).sum())})
            comparison = pd.read_csv(os.path.join(tmp, "model_comparison.csv"))
            self.assertEqual(comparison["Model"].tolist(), main.MODEL_ORDER)

    def test_refuses_inputs_that_contain_outcome_fields(self):
        df = make_df()
        df["PEINHOME"] = 1
        train, test = main.temporal_split(df)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(AssertionError):
            main.run_experiment("bad", train, test, tmp, "bad", features=main.FEATURES + ["PEINHOME"], cv=2,
                                quick=True, verbose=False)


class TestMainEntry(unittest.TestCase):

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("data")

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_runs_both_experiments_from_existing_predictor_file(self):
        make_df().to_csv(main.PREDICTOR_FILE, index=False)
        with mock.patch("builtins.print"), mock.patch.object(main.dataPrep, "main") as prep:
            summary = main.main(experiment="both", cv=2, quick=True, outdir="outputs", verbose=False)
        prep.assert_not_called()
        self.assertEqual(set(summary), {"A_pooled", "B_temporal"})
        self.assertTrue(os.path.exists("outputs/A_pooled/model_comparison.csv"))
        self.assertTrue(os.path.exists("outputs/B_temporal/model_comparison.csv"))

    def test_stops_when_no_data_available(self):
        with mock.patch("builtins.print"), mock.patch.object(main.dataPrep, "main", return_value=None) as prep:
            self.assertIsNone(main.main(verbose=False))
        prep.assert_called_once()


if __name__ == "__main__":
    unittest.main()