# Title: test_dataPrep.py
# Description: Unit tests for dataPrep.py.
# Run with: python -m unittest test_dataPrep -v
#
# Every test runs inside a temporary directory, because feature_selection() and
# main() read/write hard-coded "data/..." paths relative to the working directory.
# This keeps the real data/ folder untouched.

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import dataPrep


class TempDirTestCase(unittest.TestCase):
    """Runs each test in a fresh temp directory containing an empty data/ folder."""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("data")

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()


def make_row(**overrides):
    """Build one census-style row with every disability flag set to 2 (= no)."""
    row = {
        "PRTAGE": 70, "PRDISFLG": 1,
        "PEDISEAR": 2, "PEDISEYE": 2, "PEDISREM": 2,
        "PEDISPHY": 2, "PEDISDRS": 2, "PEDISOUT": 2,
        "PESEX": 1, "PTDTRACE": 1, "PEMARITL": 1, "HEFAMINC": 5, "PEEDUCA": 39,
        "HETENURE": 1, "GESTFIPS": 1, "PEMLR": 5,
        "HRHHID": 12345,  # not a predictor; should be dropped
    }
    row.update(overrides)
    return row


class TestConcatCsvFiles(TempDirTestCase):

    def test_concatenates_rows_in_order(self):
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv("f1.csv", index=False)
        pd.DataFrame({"A": [5], "B": [6]}).to_csv("f2.csv", index=False)

        dataPrep.concat_csv_files(["f1.csv", "f2.csv"], "out.csv")

        result = pd.read_csv("out.csv")
        self.assertEqual(result["A"].tolist(), [1, 2, 5])
        self.assertEqual(result["B"].tolist(), [3, 4, 6])

    def test_output_has_no_index_column(self):
        pd.DataFrame({"A": [1]}).to_csv("f1.csv", index=False)
        dataPrep.concat_csv_files(["f1.csv"], "out.csv")
        self.assertEqual(list(pd.read_csv("out.csv").columns), ["A"])

    def test_mismatched_columns_are_unioned(self):
        # Census years don't always share identical columns.
        pd.DataFrame({"A": [1], "B": [2]}).to_csv("f1.csv", index=False)
        pd.DataFrame({"A": [3], "C": [4]}).to_csv("f2.csv", index=False)

        dataPrep.concat_csv_files(["f1.csv", "f2.csv"], "out.csv")

        result = pd.read_csv("out.csv")
        self.assertEqual(set(result.columns), {"A", "B", "C"})
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.loc[0, "C"]))
        self.assertTrue(pd.isna(result.loc[1, "B"]))

    def test_2019_state_column_renamed_to_match_later_years(self):
        pd.DataFrame({"GCFIP": [1]}).to_csv("nov19pub.csv", index=False)
        pd.DataFrame({"GESTFIPS": [6]}).to_csv("nov21pub.csv", index=False)

        dataPrep.concat_csv_files(["nov19pub.csv", "nov21pub.csv"], "out.csv")

        result = pd.read_csv("out.csv")
        self.assertEqual(list(result.columns), ["GESTFIPS"])
        self.assertEqual(result["GESTFIPS"].tolist(), [1, 6])

    def test_empty_file_list_raises(self):
        with self.assertRaises(ValueError):
            dataPrep.concat_csv_files([], "out.csv")


class TestLoadData(TempDirTestCase):

    def test_loads_csv_into_dataframe(self):
        expected = pd.DataFrame({"PRTAGE": [65, 80], "PRDISFLG": [1, 2]})
        expected.to_csv("in.csv", index=False)

        result = dataPrep.load_data("in.csv")

        pd.testing.assert_frame_equal(result, expected)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            dataPrep.load_data("does_not_exist.csv")


class TestFilterData(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "PRTAGE":   [40, 65, 70, 90, 66],
            "PRDISFLG": [1,  1,  2,  1,  -1],
        })

    def test_age_and_disability_filter(self):
        df = self.df
        result = dataPrep.filter_data(df, [df["PRTAGE"] >= 65, df["PRDISFLG"] == 1])
        self.assertEqual(result["PRTAGE"].tolist(), [65, 90])

    def test_preserves_original_index(self):
        df = self.df
        result = dataPrep.filter_data(df, [df["PRTAGE"] >= 65, df["PRDISFLG"] == 1])
        self.assertEqual(result.index.tolist(), [1, 3])

    def test_empty_condition_list_returns_all_rows(self):
        result = dataPrep.filter_data(self.df, [])
        pd.testing.assert_frame_equal(result, self.df)

    def test_no_matches_returns_empty_frame_with_same_columns(self):
        df = self.df
        result = dataPrep.filter_data(df, [df["PRTAGE"] > 200])
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), list(df.columns))

    def test_returns_copy_not_view(self):
        df = self.df
        result = dataPrep.filter_data(df, [df["PRTAGE"] >= 65])
        result.loc[:, "PRTAGE"] = 0
        self.assertEqual(df["PRTAGE"].tolist(), [40, 65, 70, 90, 66])

    def test_does_not_modify_input(self):
        df = self.df
        before = df.copy()
        dataPrep.filter_data(df, [df["PRTAGE"] >= 65])
        pd.testing.assert_frame_equal(df, before)


INDICATOR_COLUMNS = {"DIS_HEARING", "DIS_VISION", "DIS_COGNITIVE",
                     "DIS_PHYSICAL", "DIS_SELFCARE", "DIS_INDLIVING"}


class TestFeatureSelection(TempDirTestCase):

    def test_maps_each_disability_flag_to_type(self):
        cases = {
            "PEDISEAR": "Hearing",
            "PEDISEYE": "Vision",
            "PEDISREM": "Cognitive",
            "PEDISPHY": "Physical",
            "PEDISDRS": "Self-Care",
            "PEDISOUT": "Independent Living",
        }
        df = pd.DataFrame([make_row(**{col: 1}) for col in cases])

        result = dataPrep.feature_selection(df)

        self.assertEqual(result["DISTYPE"].tolist(), list(cases.values()))

    def test_no_flags_set_gives_none(self):
        result = dataPrep.feature_selection(pd.DataFrame([make_row()]))
        self.assertEqual(result["DISTYPE"].tolist(), ["None"])

    def test_multiple_flags_lists_every_type(self):
        df = pd.DataFrame([make_row(PEDISPHY=1, PEDISEAR=1, PEDISOUT=1)])
        result = dataPrep.feature_selection(df)
        self.assertEqual(result["DISTYPE"].tolist(), ["Hearing|Physical|Independent Living"])

    def test_indicator_columns_match_flags(self):
        df = pd.DataFrame([
            make_row(PEDISEAR=1, PEDISPHY=1),
            make_row(PEDISEYE=1),
            make_row(PEDISREM=-1),  # -1 = not in universe, not a "yes"
        ])
        result = dataPrep.feature_selection(df)
        self.assertEqual(result["DIS_HEARING"].tolist(), [1, 0, 0])
        self.assertEqual(result["DIS_PHYSICAL"].tolist(), [1, 0, 0])
        self.assertEqual(result["DIS_VISION"].tolist(), [0, 1, 0])
        self.assertEqual(result["DIS_COGNITIVE"].tolist(), [0, 0, 0])
        self.assertEqual(result["DISTYPE"].tolist(), ["Hearing|Physical", "Vision", "None"])

    def test_distype_splits_back_into_indicators(self):
        df = pd.DataFrame([make_row(PEDISEAR=1, PEDISDRS=1), make_row(PEDISEYE=1)])
        result = dataPrep.feature_selection(df)
        dummies = result["DISTYPE"].str.get_dummies("|")
        self.assertEqual(dummies["Hearing"].tolist(), result["DIS_HEARING"].tolist())
        self.assertEqual(dummies["Self-Care"].tolist(), result["DIS_SELFCARE"].tolist())
        self.assertEqual(dummies["Vision"].tolist(), result["DIS_VISION"].tolist())

    def test_missing_disability_columns_treated_as_not_set(self):
        df = pd.DataFrame({"PRTAGE": [70], "PEDISEYE": [1]})
        result = dataPrep.feature_selection(df)
        self.assertEqual(result["DISTYPE"].tolist(), ["Vision"])

    def test_keeps_only_predictor_columns(self):
        result = dataPrep.feature_selection(pd.DataFrame([make_row()]))
        self.assertEqual(
            set(result.columns),
            {"PRTAGE", "PESEX", "PTDTRACE", "PEMARITL", "HEFAMINC", "PEEDUCA",
             "HETENURE", "GESTFIPS", "PEMLR", "DISTYPE"} | INDICATOR_COLUMNS,
        )

    def test_missing_predictor_columns_are_not_added(self):
        df = pd.DataFrame({"PRTAGE": [70], "PEDISEAR": [1], "HRHHID": [1]})
        result = dataPrep.feature_selection(df)
        self.assertEqual(set(result.columns), {"PRTAGE", "DISTYPE"} | INDICATOR_COLUMNS)
        self.assertEqual(result["DIS_HEARING"].tolist(), [1])
        self.assertEqual(result["DIS_VISION"].tolist(), [0])

    def test_writes_filtered_and_predictor_csvs(self):
        df = pd.DataFrame([make_row(PEDISEAR=1)])

        result = dataPrep.feature_selection(df)

        filtered = pd.read_csv("data/filtered_data.csv")
        self.assertIn("DISTYPE", filtered.columns)
        self.assertIn("HRHHID", filtered.columns)  # filtered keeps all columns

        predictor = pd.read_csv("data/predictor_data.csv")
        self.assertEqual(set(predictor.columns), set(result.columns))
        self.assertEqual(predictor["DISTYPE"].tolist(), ["Hearing"])


class TestMain(TempDirTestCase):

    def test_no_csv_files_returns_none(self):
        with mock.patch("builtins.print") as mock_print:
            result = dataPrep.main()
        self.assertIsNone(result)
        mock_print.assert_called_once_with(
            "No raw census files (nov*pub.csv) were found in the data/ folder.")

    def test_ignores_non_census_csvs(self):
        pd.DataFrame([make_row(PRTAGE=70)]).to_csv("data/nov23pub.csv", index=False)
        pd.DataFrame([make_row(PRTAGE=99)]).to_csv("data/other.csv", index=False)

        with mock.patch("builtins.print"):
            result = dataPrep.main()

        self.assertEqual(result["PRTAGE"].tolist(), [70])

    def test_rerun_does_not_reingest_its_own_outputs(self):
        pd.DataFrame([
            make_row(PRTAGE=70, PEDISEYE=1),
            make_row(PRTAGE=80, PEDISPHY=1),
        ]).to_csv("data/nov23pub.csv", index=False)

        with mock.patch("builtins.print"):
            first = dataPrep.main()
            second = dataPrep.main()

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(pd.read_csv("data/filtered_data.csv")), 2)

    def test_end_to_end_filters_and_selects_features(self):
        pd.DataFrame([
            make_row(PRTAGE=70, PRDISFLG=1, PEDISEYE=1),  # kept
            make_row(PRTAGE=40, PRDISFLG=1, PEDISEAR=1),  # too young
        ]).to_csv("data/nov19pub.csv", index=False)
        pd.DataFrame([
            make_row(PRTAGE=80, PRDISFLG=1, PEDISPHY=1),  # kept
            make_row(PRTAGE=85, PRDISFLG=2),              # no disability
        ]).to_csv("data/nov21pub.csv", index=False)

        with mock.patch("builtins.print"):
            result = dataPrep.main()

        self.assertEqual(result["PRTAGE"].tolist(), [70, 80])
        self.assertEqual(result["DISTYPE"].tolist(), ["Vision", "Physical"])
        self.assertNotIn("PRDISFLG", result.columns)
        self.assertTrue(os.path.exists("data/combined_data.csv"))
        self.assertTrue(os.path.exists("data/filtered_data.csv"))
        self.assertTrue(os.path.exists("data/predictor_data.csv"))
        self.assertEqual(len(pd.read_csv("data/combined_data.csv")), 4)  # raw rows, not overwritten


if __name__ == "__main__":
    unittest.main()
