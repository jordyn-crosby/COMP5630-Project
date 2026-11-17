# Title: test_plot_results.py
# Description: Unit tests for plot_results.py. Run with: python -m unittest test_plot_results -v
#
# Tests write small synthetic model_comparison.csv files into a temporary outputs/ folder, so the real
# outputs/ and results/ folders are untouched.

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import plot_results

MODELS = ["Dummy (majority)", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def make_comparison(seed=0):
    rows = []
    for i, model in enumerate(MODELS):
        tn, fp, fn, tp = (60, 0, 40, 0) if model.startswith("Dummy") else (45 + i, 15 - i, 12, 28 + seed)
        n = tn + fp + fn + tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"Model": model, "Accuracy": (tn + tp) / n, "Balanced Accuracy": 0.5 if tp == 0 else 0.68,
                     "Precision (excluded)": prec, "Recall (excluded)": rec,
                     "F1 (excluded)": 0.0 if tp == 0 else 2 * prec * rec / (prec + rec),
                     "Macro F1": 0.37 if tp == 0 else 0.6 + 0.01 * i, "ROC AUC": 0.5 if tp == 0 else 0.7 + 0.01 * i,
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    return pd.DataFrame(rows)


class TestPlotResults(unittest.TestCase):

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def write_experiments(self, folders):
        for k, folder in enumerate(folders):
            os.makedirs(os.path.join("outputs", folder), exist_ok=True)
            make_comparison(seed=k).to_csv(os.path.join("outputs", folder, "model_comparison.csv"), index=False)

    def test_writes_table_and_both_figures(self):
        self.write_experiments(["A_pooled", "B_temporal", os.path.join("H3_pooled", "reduced"), "C_accuracy_tuned"])
        with mock.patch("builtins.print"):
            paths = plot_results.main()
        for p in paths.values():
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 1_000)
        table = pd.read_csv(paths["table"])
        self.assertEqual(len(table), 4 * len(MODELS))
        self.assertEqual(table["Experiment"].iloc[0], "Pooled 70/30 (household-grouped)")
        self.assertIn("Baseline (always connected)", set(table["Model"]))
        self.assertNotIn("Dummy (majority)", set(table["Model"]))
        counts = table[["Correct Connected (TN)", "False Alarms (FP)", "Missed Excluded (FN)", "Correct Excluded (TP)"]]
        self.assertTrue((counts.sum(axis=1) == table["Test Size"]).all())

    def test_skips_missing_experiments(self):
        self.write_experiments(["A_pooled"])
        with mock.patch("builtins.print"):
            paths = plot_results.main()
        table = pd.read_csv(paths["table"])
        self.assertEqual(set(table["Experiment"]), {"Pooled 70/30 (household-grouped)"})
        self.assertTrue(os.path.exists(paths["confusion"]))

    def test_custom_results_folder(self):
        self.write_experiments(["B_temporal"])
        with mock.patch("builtins.print"):
            paths = plot_results.main(results_dir="figs")
        self.assertTrue(all(p.startswith("figs") for p in paths.values()))

    def test_no_outputs_returns_none(self):
        with mock.patch("builtins.print") as mock_print:
            self.assertIsNone(plot_results.main())
        mock_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
