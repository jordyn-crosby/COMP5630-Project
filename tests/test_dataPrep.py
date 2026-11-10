# Title: test_dataPrep.py
# Description: Unit tests for dataPrep.py (cohort rule, outcome rule, feature exclusions, outputs).
# Run with: python -m unittest test_dataPrep -v
#
# Every file-writing test runs inside a temporary directory, because feature_selection() and main()
# read/write hard-coded "data/..." paths relative to the working directory.

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import dataPrep


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.makedirs("data")

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()


def make_row(**overrides):
    """One census-style cohort row: age 70, mobility difficulty, connected household, uses Internet at home."""
    row = {
        "HRYEAR4": 2023, "HRHHID": 100, "HRHHID2": 1, "PULINENO": 1, "HRNUMHOU": 2, "PWSSWGT": 1000.0,
        "PRTAGE": 70, "PRDISFLG": 1,
        "PEDISEAR": 2, "PEDISEYE": 2, "PEDISREM": 2, "PEDISPHY": 1, "PEDISDRS": 2, "PEDISOUT": 2,
        "PESEX": 1, "PTDTRACE": 1, "PEMARITL": 1, "HEFAMINC": 5, "PEEDUCA": 39, "HETENURE": 1,
        "GESTFIPS": 1, "PEMLR": 5, "HXFAMINC": 0,
        "HEINHOME": 1, "PEINHOME": 1, "PEINWORK": -1, "PEINSCHL": 2, "PEINCAFE": 2, "PEINTRAV": 2,
        "PEINLICO": 2, "PEINELHO": 2, "PEINOTHR": 2, "HENETCHK": -1, "PENETCK2": -1, "HEEVRHOM": -1,
    }
    row.update(overrides)
    return row


def nonuser(**overrides):
    """Connected household, but this person is reported nowhere as a user and the follow-up confirms it."""
    base = make_row(PEINHOME=2, HENETCHK=2, PENETCK2=2)
    base.update(overrides)
    return base


# ----------------------------------------------------------------------------------------------------
# Loading and harmonizing
# ----------------------------------------------------------------------------------------------------

class TestLoadRaw(TempDirTestCase):

    def test_2019_state_column_renamed_and_only_needed_columns_read(self):
        row = make_row()
        row["GCFIP"] = row.pop("GESTFIPS")
        row["UNRELATED"] = 99
        pd.DataFrame([row]).to_csv("data/CPS2019.csv", index=False)

        df = dataPrep.load_raw("data/CPS2019.csv")

        self.assertIn("GESTFIPS", df.columns)
        self.assertNotIn("GCFIP", df.columns)
        self.assertNotIn("UNRELATED", df.columns)
        self.assertEqual(set(df.columns), set(dataPrep.RAW_COLUMNS))

    def test_missing_required_column_raises(self):
        row = make_row()
        del row["PENETCK2"]
        pd.DataFrame([row]).to_csv("data/CPS2023.csv", index=False)
        with self.assertRaises(ValueError):
            dataPrep.load_raw("data/CPS2023.csv")

    def test_concat_unions_years(self):
        pd.DataFrame([make_row(HRYEAR4=2021)]).to_csv("data/CPS2021.csv", index=False)
        pd.DataFrame([make_row(HRYEAR4=2023)]).to_csv("data/CPS2023.csv", index=False)
        df = dataPrep.concat_csv_files(["data/CPS2021.csv", "data/CPS2023.csv"])
        self.assertEqual(df["HRYEAR4"].tolist(), [2021, 2023])

    def test_empty_file_list_raises(self):
        with self.assertRaises(ValueError):
            dataPrep.concat_csv_files([])

    def test_find_raw_files_accepts_both_naming_schemes(self):
        for name in ["CPS2019.csv", "nov21pub.csv", "predictor_data.csv", "other.csv"]:
            pd.DataFrame([make_row()]).to_csv(f"data/{name}", index=False)
        found = [os.path.basename(f) for f in dataPrep.find_raw_files()]
        self.assertEqual(found, ["CPS2019.csv", "nov21pub.csv"])


# ----------------------------------------------------------------------------------------------------
# Cohort rule: age 65+ and at least one of the six CPS disability categories
# ----------------------------------------------------------------------------------------------------

class TestCohort(unittest.TestCase):

    def test_filter_data_applies_every_condition(self):
        df = pd.DataFrame({"PRTAGE": [40, 65, 70, 90], "PRDISFLG": [1, 1, 2, 1]})
        result = dataPrep.filter_data(df, [df["PRTAGE"] >= 65, df["PRDISFLG"] == 1])
        self.assertEqual(result["PRTAGE"].tolist(), [65, 90])

    def test_age_boundary(self):
        df = pd.DataFrame([make_row(PRTAGE=64), make_row(PRTAGE=65)])
        self.assertEqual(dataPrep.apply_cohort(df)["PRTAGE"].tolist(), [65])

    def test_each_of_six_categories_qualifies(self):
        rows = []
        for flag in dataPrep.DISABILITY_FLAGS:
            r = make_row(PEDISPHY=2)
            r[flag] = 1
            rows.append(r)
        self.assertEqual(len(dataPrep.DISABILITY_FLAGS), 6)
        self.assertEqual(len(dataPrep.apply_cohort(pd.DataFrame(rows))), 6)

    def test_independent_living_alone_qualifies(self):
        # PEDISOUT = 1 with none of the other five categories is in the six-category cohort
        df = pd.DataFrame([make_row(PEDISPHY=2, PEDISOUT=1, PRDISFLG=1)])
        result = dataPrep.add_disability_indicators(dataPrep.apply_cohort(df))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["DISTYPE"].tolist(), ["Independent Living"])
        self.assertEqual(result["DIS_INDLIVING"].tolist(), [1])

    def test_no_disability_does_not_qualify(self):
        df = pd.DataFrame([make_row(PEDISPHY=2, PRDISFLG=2)])
        self.assertEqual(len(dataPrep.apply_cohort(df)), 0)

    def test_overlap_is_preserved_not_deduplicated(self):
        df = pd.DataFrame([make_row(PEDISEAR=1, PEDISPHY=1, PEDISREM=1)])
        result = dataPrep.add_disability_indicators(dataPrep.apply_cohort(df))
        self.assertEqual(result[dataPrep.DISABILITY_INDICATORS].sum(axis=1).tolist(), [3])
        self.assertEqual(result["DISTYPE"].tolist(), ["Hearing|Cognitive|Mobility"])


# ----------------------------------------------------------------------------------------------------
# Outcome rule: the five validation cases from the completion roadmap, plus audits
# ----------------------------------------------------------------------------------------------------

class TestOutcomeRule(unittest.TestCase):

    def outcome(self, **overrides):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(**overrides)]))
        return df["DIGITAL_EXCLUSION"].iloc[0], df["OUTCOME_STATUS"].iloc[0]

    def test_case1_connected_household_with_non_using_individual_is_excluded(self):
        value, status = self.outcome(HEINHOME=1, PEINHOME=2, HENETCHK=2, PENETCK2=2)
        self.assertEqual((value, status), (1, "resolved"))

    def test_case2_individual_who_uses_internet_away_from_home_is_not_excluded(self):
        value, status = self.outcome(HEINHOME=2, PEINHOME=2, PEINLICO=1, HEEVRHOM=2)
        self.assertEqual((value, status), (0, "resolved"))

    def test_case3_household_with_no_home_internet_and_no_other_use_is_excluded(self):
        value, status = self.outcome(HEINHOME=2, PEINHOME=2, HENETCHK=2, PENETCK2=2, HEEVRHOM=2)
        self.assertEqual((value, status), (1, "resolved"))

    def test_case4_followup_that_corrects_apparent_non_use_is_not_excluded(self):
        # named as a user in the NETCHK follow-up even though no location flag was set
        value, status = self.outcome(HEINHOME=1, PEINHOME=2, HENETCHK=1, PENETCK2=1)
        self.assertEqual((value, status), (0, "resolved"))

    def test_case5_unresolved_responses_are_flagged_and_left_without_outcome(self):
        for code in (-2, -3, -9):
            value, status = self.outcome(HEINHOME=1, PEINHOME=2, HENETCHK=code, PENETCK2=-1)
            self.assertTrue(np.isnan(value))
            self.assertEqual(status, "unresolved")

    def test_a_yes_anywhere_overrides_unresolved_codes(self):
        value, status = self.outcome(PEINHOME=1, HENETCHK=-2)
        self.assertEqual((value, status), (0, "resolved"))

    def test_single_person_household_non_user_has_no_followup_and_is_excluded(self):
        value, status = self.outcome(HRNUMHOU=1, HEINHOME=2, PEINHOME=2, HENETCHK=-1, PENETCK2=-1, HEEVRHOM=2)
        self.assertEqual((value, status), (1, "resolved"))

    def test_home_only_rule_would_differ_from_composite(self):
        # a home-only variable would call this person excluded; the composite correctly does not
        df = dataPrep.build_outcome(pd.DataFrame([make_row(HEINHOME=2, PEINHOME=2, PEINELHO=1)]))
        self.assertEqual(df["DIGITAL_EXCLUSION"].iloc[0], 0)
        self.assertEqual(df["HEINHOME"].iloc[0], 2)

    def test_audit_rejects_home_user_in_household_without_home_internet(self):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(HEINHOME=2, PEINHOME=1)]))
        with self.assertRaises(AssertionError):
            dataPrep.audit_outcome(df)

    def test_audit_reports_counts(self):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(), nonuser(), make_row(PEINHOME=2, HENETCHK=1, PENETCK2=1)]))
        checks = dataPrep.audit_outcome(df)
        self.assertEqual(checks["followup_named_users"], 1)
        self.assertEqual(checks["followup_named_users_counted_as_users"], 1)
        self.assertEqual(checks["unresolved_records"], 0)
        self.assertEqual(checks["cohort_rows_without_prdisflg"], 0)


# ----------------------------------------------------------------------------------------------------
# Feature selection: inputs, exclusions, outputs
# ----------------------------------------------------------------------------------------------------

class TestFeatureSelection(TempDirTestCase):

    def test_modeling_table_has_inputs_outcome_metadata_and_no_internet_fields(self):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(), nonuser(PULINENO=2)]))
        result = dataPrep.feature_selection(df)

        expected = set(dataPrep.METADATA_COLUMNS + dataPrep.PREDICTOR_COLUMNS + dataPrep.DISABILITY_INDICATORS
                       + ["DISTYPE", "DIGITAL_EXCLUSION"])
        self.assertEqual(set(result.columns), expected)
        for col in dataPrep.OUTCOME_SOURCE_COLUMNS + ["INTERNET_USER", "OUTCOME_STATUS", "PRDISFLG"] + dataPrep.DISABILITY_FLAGS:
            self.assertNotIn(col, result.columns)
        self.assertEqual(len(dataPrep.DISABILITY_INDICATORS), 6)
        self.assertEqual(result["DIGITAL_EXCLUSION"].tolist(), [0, 1])
        self.assertEqual(result["DIGITAL_EXCLUSION"].dtype.kind, "i")

    def test_indicator_columns_match_flags(self):
        df = dataPrep.build_outcome(pd.DataFrame([
            make_row(PEDISEAR=1, PEDISPHY=1), make_row(PEDISEYE=1, PEDISPHY=2), make_row(PEDISREM=-1, PEDISPHY=1)]))
        result = dataPrep.feature_selection(df)
        self.assertEqual(result["DIS_HEARING"].tolist(), [1, 0, 0])
        self.assertEqual(result["DIS_MOBILITY"].tolist(), [1, 0, 1])
        self.assertEqual(result["DIS_VISION"].tolist(), [0, 1, 0])
        self.assertEqual(result["DIS_COGNITIVE"].tolist(), [0, 0, 0])
        self.assertEqual(result["DISTYPE"].tolist(), ["Hearing|Mobility", "Vision", "Mobility"])
        self.assertEqual(result["DIS_INDLIVING"].tolist(), [0, 0, 0])

    def test_unresolved_rows_are_dropped_from_modeling_table_but_kept_in_audit_table(self):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(), nonuser(HENETCHK=-3, PULINENO=2)]))
        result = dataPrep.feature_selection(df)
        self.assertEqual(len(result), 1)
        filtered = pd.read_csv("data/filtered_data.csv")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered["OUTCOME_STATUS"].tolist(), ["resolved", "unresolved"])
        self.assertIn("HEINHOME", filtered.columns)   # audit table keeps the outcome source fields
        self.assertIn("PENETCK2", filtered.columns)

    def test_writes_predictor_csv_matching_returned_frame(self):
        df = dataPrep.build_outcome(pd.DataFrame([make_row(), nonuser(PULINENO=2)]))
        result = dataPrep.feature_selection(df)
        written = pd.read_csv("data/predictor_data.csv")
        self.assertEqual(set(written.columns), set(result.columns))
        self.assertEqual(written["DIGITAL_EXCLUSION"].tolist(), [0, 1])


# ----------------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------------

class TestMain(TempDirTestCase):

    def test_no_csv_files_returns_none(self):
        with mock.patch("builtins.print") as mock_print:
            result = dataPrep.main()
        self.assertIsNone(result)
        mock_print.assert_called_once()

    def test_end_to_end_cohort_outcome_and_summary(self):
        r2019 = make_row(HRYEAR4=2019)
        r2019["GCFIP"] = r2019.pop("GESTFIPS")
        pd.DataFrame([
            r2019,                                                     # 2019 user, kept
            {**make_row(HRYEAR4=2019, PRTAGE=40), "GCFIP": 13},        # too young
        ]).drop(columns=["GESTFIPS"], errors="ignore").to_csv("data/CPS2019.csv", index=False)
        pd.DataFrame([
            nonuser(HRYEAR4=2023, HRHHID=200),                         # excluded, kept
            make_row(HRYEAR4=2023, HRHHID=201, PEDISPHY=2, PEDISOUT=1),  # independent living only: in cohort, user
            make_row(HRYEAR4=2023, HRHHID=203, PEDISPHY=2, PRDISFLG=2),  # no disability: not in cohort
            nonuser(HRYEAR4=2023, HRHHID=202, HENETCHK=-2, PENETCK2=-1),  # unresolved: dropped from modeling
        ]).to_csv("data/CPS2023.csv", index=False)

        with mock.patch("builtins.print"):
            result = dataPrep.main()

        self.assertEqual(result["HRYEAR4"].tolist(), [2019, 2023, 2023])
        self.assertEqual(result["DIGITAL_EXCLUSION"].tolist(), [0, 1, 0])
        self.assertEqual(result["DISTYPE"].tolist(), ["Mobility", "Mobility", "Independent Living"])
        self.assertEqual(result["GESTFIPS"].tolist(), [1, 1, 1])
        self.assertNotIn("PEINHOME", result.columns)

        summary = pd.read_csv("data/cohort_summary.csv")
        by_year = summary.set_index("year")
        self.assertEqual(int(by_year.loc["2019", "cohort_six_categories"]), 1)
        self.assertEqual(int(by_year.loc["2023", "cohort_six_categories"]), 3)
        self.assertEqual(int(by_year.loc["2023", "unresolved_dropped"]), 1)
        self.assertEqual(int(by_year.loc["2023", "modeling_n"]), 2)
        self.assertEqual(int(by_year.loc["total", "modeling_n"]), 3)
        self.assertTrue(os.path.exists("data/filtered_data.csv"))
        self.assertTrue(os.path.exists("data/predictor_data.csv"))


if __name__ == "__main__":
    unittest.main()