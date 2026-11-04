# Title: test_plot_results.py
# Description: Unit tests for plot_results.py.
# Run with: python -m unittest test_plot_results -v
#
# Tests write a small synthetic results table into a temporary directory, so the real results/ folder is untouched.

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import plot_results

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Baseline (always connected)"]


def make_results(experiments=("Random split", "Train 2019+2021, test 2023", "Reduced (4 predictors)")):
    rows = []
    for e, experiment in enumerate(experiments):
        for m, model in enumerate(MODELS):
            baseline = model.startswith("Baseline")
            tn, fp, fn, tp = (60, 0, 40, 0) if baseline else (45 + m, 15 - m, 12, 28 + e)
            total = tn + fp + fn + tp
            rows.append({
                "Experiment": experiment, "Model": model, "Test Size": total,
                "Accuracy": (tn + tp) / total, "Balanced Accuracy": 0.5 if baseline else 0.68,
                "Precision": 0.0 if baseline else 0.6, "Recall": 0.0 if baseline else 0.7,
                "F1 (excluded)": 0.0 if baseline else 0.64,
                "Macro F1": 0.37 if baseline else 0.6 + 0.01 * m,
                "ROC AUC": 0.5 if baseline else 0.7 + 0.01 * m,
                "Correct Connected (TN)": tn, "False Alarms (FP)": fp,
                "Missed Excluded (FN)": fn, "Correct Excluded (TP)": tp,
            })
    return pd.DataFrame(rows)


class TestPlotResults(unittest.TestCase):

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("results")

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_writes_both_charts(self):
        make_results().to_csv(plot_results.RESULTS_FILE, index=False)
        with mock.patch("builtins.print"):
            paths = plot_results.main()
        for path in paths.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 10_000)

    def test_works_with_a_single_experiment(self):
        make_results(experiments=("Random split",)).to_csv(plot_results.RESULTS_FILE, index=False)
        with mock.patch("builtins.print"):
            paths = plot_results.main()
        self.assertTrue(all(os.path.exists(p) for p in paths.values()))

    def test_missing_results_file_returns_none(self):
        with mock.patch("builtins.print") as mock_print:
            self.assertIsNone(plot_results.main())
        mock_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
