# Title: dataPrep.py
# Description: Prepares the modeling table for the COMP 6630 project
# "Who Gets Left Behind? Machine Learning for Digital Exclusion Among Older Adults with Disabilities".
#
# Phase 1: Data preparation
#   Read the November 2019, 2021 and 2023 CPS Computer and Internet Use Supplement public-use files,
#   harmonize column names across years, and restrict to the study cohort:
#   adults aged 65+ who report at least one of the six CPS disability categories
#   (hearing, vision, cognitive, mobility, dressing/bathing, independent living).
#   Team decision of November 4, 2026: all six categories (the proposal text named five; the team chose
#   the full CPS set). An OR across the six flags equals the Census any-disability flag PRDISFLG = 1.
#
# Phase 2: Outcome construction and feature selection
#   Build the single binary outcome DIGITAL_EXCLUSION from the personal and household internet-use
#   questions BEFORE any source field is dropped, then keep the ten predictor groups (15 raw input
#   columns), the outcome, and evaluation metadata (year, household id, person line, survey weight).
#   Internet-use and device fields are kept only in data/filtered_data.csv for auditing and are never
#   written to data/predictor_data.csv, so they cannot leak into the models.
#
# Outputs (all written to data/):
#   filtered_data.csv   cohort rows with predictors, outcome, outcome source fields and metadata (audit table)
#   predictor_data.csv  modeling table: metadata + 15 inputs + DIGITAL_EXCLUSION (no internet fields)
#   cohort_summary.csv  per-year and total counts at every cohort step
#
# Run:  python dataPrep.py            (expects data/CPS2019.csv, data/CPS2021.csv, data/CPS2023.csv,
#                                      or the Census names data/nov19pub.csv, nov21pub.csv, nov23pub.csv)
#       python dataPrep.py --raw-dir "../"   (raw files live elsewhere; outputs still go to data/)

import argparse
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
FILTERED_FILE = os.path.join(DATA_DIR, "filtered_data.csv")
PREDICTOR_FILE = os.path.join(DATA_DIR, "predictor_data.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "cohort_summary.csv")

# raw files: the course copies (CPS2019.csv ...) or the Census download names (nov19pub.csv ...)
RAW_FILE_GLOBS = ["CPS20*.csv", "nov*pub.csv"]

# columns whose names differ between survey years, mapped to a common name
# (2019 stores state FIPS as GCFIP; 2021 and 2023 use GESTFIPS)
COLUMN_RENAMES = {"GCFIP": "GESTFIPS"}

MIN_AGE = 65

# ---------------------------------------------------------------------------------------------------
# Column dictionary
# ---------------------------------------------------------------------------------------------------

# evaluation metadata: never used as model inputs
METADATA_COLUMNS = [
    "HRYEAR4",   # survey year
    "HRHHID",    # household identifier (part 1)
    "HRHHID2",   # household identifier (part 2)
    "PULINENO",  # person line number within household
    "HRNUMHOU",  # number of household members (used to audit the NETCHK skip pattern)
    "PWSSWGT",   # person-level supplement weight (descriptive reporting only)
]

# nine demographic / socioeconomic predictors (nine of the ten predictor groups)
PREDICTOR_COLUMNS = [
    "PRTAGE",    # age (top-coded 80 = 80 to 84, 85 = 85 and older)
    "PESEX",     # sex
    "PTDTRACE",  # race
    "PEEDUCA",   # education (highest level completed)
    "HEFAMINC",  # family income bracket (Census allocates missing values; see HXFAMINC)
    "HETENURE",  # housing tenure
    "PEMARITL",  # marital status
    "PEMLR",     # employment / labor force status
    "GESTFIPS",  # state FIPS code
]

# tenth predictor group: disability type. Census flag -> (readable name, 0/1 indicator column)
# All six CPS disability categories (team decision, November 4, 2026). The cohort rule is an OR across
# these flags, which is the same population as PRDISFLG = 1; the flags also enter the model as inputs.
DISABILITY_MAP = {
    "PEDISEAR": ("Hearing", "DIS_HEARING"),
    "PEDISEYE": ("Vision", "DIS_VISION"),
    "PEDISREM": ("Cognitive", "DIS_COGNITIVE"),
    "PEDISPHY": ("Mobility", "DIS_MOBILITY"),             # difficulty walking or climbing stairs
    "PEDISDRS": ("Self-Care", "DIS_SELFCARE"),            # difficulty dressing or bathing
    "PEDISOUT": ("Independent Living", "DIS_INDLIVING"),  # difficulty doing errands alone
}
DISABILITY_FLAGS = list(DISABILITY_MAP)
DISABILITY_INDICATORS = [ind for _, ind in DISABILITY_MAP.values()]
EXTRA_DISABILITY_AUDIT = ["PRDISFLG"]  # Census any-disability flag, kept to audit the cohort rule

# person-level internet-use-by-location items (1 = this person uses the Internet at that location)
PERSON_LOCATION_COLUMNS = [
    "PEINHOME",  # at home
    "PEINWORK",  # at work
    "PEINSCHL",  # at school
    "PEINCAFE",  # at a coffee shop or other business
    "PEINTRAV",  # while traveling between places
    "PEINLICO",  # at a library, community center or other public place
    "PEINELHO",  # at someone else's home
    "PEINOTHR",  # some other location
]
# household-level items used to build and audit the outcome
HOUSEHOLD_INTERNET_COLUMNS = [
    "HEINHOME",  # anyone in household uses the Internet at home
    "HENETCHK",  # follow-up in multi-person households: do the apparent non-users use the Internet anywhere?
    "HEEVRHOM",  # household ever used the Internet from home (asked when HEINHOME != 1)
]
PERSON_FOLLOWUP_COLUMN = "PENETCK2"  # person named as an Internet user in the NETCHK follow-up (1 = yes)

OUTCOME_SOURCE_COLUMNS = PERSON_LOCATION_COLUMNS + HOUSEHOLD_INTERNET_COLUMNS + [PERSON_FOLLOWUP_COLUMN]

# Census edited-value conventions: -1 not in universe, -2 don't know, -3 refused, -9 no response
UNRESOLVED_CODES = (-2, -3, -9)

OUTCOME = "DIGITAL_EXCLUSION"

# every column read from the raw files (the raw files have 500+ columns; reading only these keeps memory low)
RAW_COLUMNS = (METADATA_COLUMNS + PREDICTOR_COLUMNS + DISABILITY_FLAGS + EXTRA_DISABILITY_AUDIT
               + OUTCOME_SOURCE_COLUMNS + ["HXFAMINC"])


# ---------------------------------------------------------------------------------------------------
# Phase 1: load and harmonize
# ---------------------------------------------------------------------------------------------------

def find_raw_files(raw_dir=DATA_DIR):
    files = []
    for pattern in RAW_FILE_GLOBS:
        files.extend(glob.glob(os.path.join(raw_dir, pattern)))
    return sorted(set(files))


def load_raw(file_path, columns=None):
    """Read one raw CPS file, keeping only the needed columns and harmonizing year-specific names."""
    columns = RAW_COLUMNS if columns is None else columns
    header = pd.read_csv(file_path, nrows=0).columns
    reverse = {new: old for old, new in COLUMN_RENAMES.items()}
    # translate a harmonized name back to the file's own spelling when needed (GESTFIPS -> GCFIP for 2019)
    wanted = []
    for col in columns:
        if col in header:
            wanted.append(col)
        elif col in reverse and reverse[col] in header:
            wanted.append(reverse[col])
    df = pd.read_csv(file_path, usecols=wanted).rename(columns=COLUMN_RENAMES)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {missing}")
    return df


def concat_csv_files(file_list, output_file=None, columns=None):
    """Concatenate raw files into one DataFrame (and optionally save it)."""
    if not file_list:
        raise ValueError("file_list is empty")
    df = pd.concat([load_raw(f, columns) for f in file_list], ignore_index=True)
    if output_file:
        df.to_csv(output_file, index=False)
    return df


def load_data(file_path):
    return pd.read_csv(file_path)


def filter_data(df, filter_list):
    """Keep rows satisfying every boolean condition in filter_list."""
    mask = pd.Series(True, index=df.index)
    for condition in filter_list:
        mask &= condition
    return df[mask].copy()


def any_named_disability(df):
    """True when at least one of the six disability flags equals 1 (overlap is preserved)."""
    present = [c for c in DISABILITY_FLAGS if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return (df[present] == 1).any(axis=1)


def apply_cohort(df):
    """Cohort rule: age 65+ AND at least one of the six CPS disability categories."""
    return filter_data(df, [df["PRTAGE"] >= MIN_AGE, any_named_disability(df)])


# ---------------------------------------------------------------------------------------------------
# Phase 2a: outcome construction (must run before any source field is dropped)
# ---------------------------------------------------------------------------------------------------

def build_outcome(df):
    """
    Add INTERNET_USER, OUTCOME_STATUS and DIGITAL_EXCLUSION to df.

    Rule (person level, see outcome_specification.md):
      INTERNET_USER = 1 if the person is reported as using the Internet at ANY of the eight locations
                      (PEINHOME ... PEINOTHR == 1) OR the household follow-up named the person as a user
                      (PENETCK2 == 1). Otherwise 0.
      OUTCOME_STATUS = "resolved" unless the person is a non-user and any outcome source field carries a
                      don't know / refused / no response code (-2, -3, -9); then "unresolved".
      DIGITAL_EXCLUSION = 1 - INTERNET_USER for resolved records; NaN for unresolved records
                      (unresolved records are counted in cohort_summary.csv and dropped from modeling).
    Household home access (HEINHOME) is used to audit the rule, not to define it: a person in a
    connected household who does not personally use the Internet anywhere is excluded; a person in a
    household without home Internet who uses it elsewhere is not.
    """
    df = df.copy()
    loc_cols = [c for c in PERSON_LOCATION_COLUMNS if c in df.columns]
    uses_somewhere = (df[loc_cols] == 1).any(axis=1) if loc_cols else pd.Series(False, index=df.index)
    followup_yes = (df[PERSON_FOLLOWUP_COLUMN] == 1) if PERSON_FOLLOWUP_COLUMN in df.columns \
        else pd.Series(False, index=df.index)
    user = uses_somewhere | followup_yes

    check_cols = [c for c in OUTCOME_SOURCE_COLUMNS if c in df.columns]
    unresolved_code = df[check_cols].isin(UNRESOLVED_CODES).any(axis=1) if check_cols \
        else pd.Series(False, index=df.index)
    # a "yes" anywhere settles the question; only non-users with unresolved codes stay unresolved
    unresolved = (~user) & unresolved_code

    df["INTERNET_USER"] = user.astype(int)
    df["OUTCOME_STATUS"] = np.where(unresolved, "unresolved", "resolved")
    df[OUTCOME] = np.where(unresolved, np.nan, (~user).astype(int)).astype(float)
    return df


def audit_outcome(df):
    """Consistency checks on real records. Returns a dict of counts; raises if a hard invariant fails."""
    checks = {}
    if {"HEINHOME", "PEINHOME"} <= set(df.columns):
        bad = int(((df["HEINHOME"] == 2) & (df["PEINHOME"] == 1)).sum())
        checks["home_user_in_household_without_home_internet"] = bad
        if bad:
            raise AssertionError(f"{bad} persons use the Internet at home in households with no home Internet")
    if PERSON_FOLLOWUP_COLUMN in df.columns:
        named = df[df[PERSON_FOLLOWUP_COLUMN] == 1]
        checks["followup_named_users"] = int(len(named))
        checks["followup_named_users_counted_as_users"] = int(named["INTERNET_USER"].sum())
    if "PRDISFLG" in df.columns:
        # the six-flag OR must reproduce the Census any-disability flag for everyone in the cohort
        checks["cohort_rows_without_prdisflg"] = int((df["PRDISFLG"] != 1).sum())
    if {"HRNUMHOU", "HENETCHK"} <= set(df.columns):
        multi_nonuser_unchecked = int(((df["HRNUMHOU"] > 1) & (df["INTERNET_USER"] == 0)
                                       & (df["HENETCHK"] == -1)).sum())
        checks["multi_person_nonusers_without_followup"] = multi_nonuser_unchecked
    checks["unresolved_records"] = int((df["OUTCOME_STATUS"] == "unresolved").sum())
    return checks


# ---------------------------------------------------------------------------------------------------
# Phase 2b: feature selection
# ---------------------------------------------------------------------------------------------------

def add_disability_indicators(df):
    """One 0/1 column per disability type plus a readable DISTYPE summary (e.g. 'Hearing|Mobility')."""
    df = df.copy()
    for flag_col, (name, indicator_col) in DISABILITY_MAP.items():
        df[indicator_col] = (df[flag_col] == 1).astype(int) if flag_col in df.columns else 0
    names = pd.Series([name + "|" for name, _ in DISABILITY_MAP.values()], index=DISABILITY_INDICATORS)
    df["DISTYPE"] = df[DISABILITY_INDICATORS].astype(bool).dot(names).str.rstrip("|").replace("", "None")
    return df


def feature_selection(df, write=True):
    """
    Build the modeling table. Expects a cohort DataFrame that already contains the outcome columns.
    Writes data/filtered_data.csv (audit table) and data/predictor_data.csv (modeling table).
    """
    df = add_disability_indicators(df)

    audit_columns = [c for c in METADATA_COLUMNS + PREDICTOR_COLUMNS + DISABILITY_FLAGS + EXTRA_DISABILITY_AUDIT
                     + DISABILITY_INDICATORS + ["DISTYPE", "HXFAMINC"] + OUTCOME_SOURCE_COLUMNS
                     + ["INTERNET_USER", "OUTCOME_STATUS", OUTCOME] if c in df.columns]
    filtered = df[audit_columns]

    model_columns = [c for c in METADATA_COLUMNS + PREDICTOR_COLUMNS + DISABILITY_INDICATORS
                     + ["DISTYPE", OUTCOME] if c in df.columns]
    predictor = df.loc[df["OUTCOME_STATUS"] == "resolved", model_columns].copy()
    predictor[OUTCOME] = predictor[OUTCOME].astype(int)

    leaked = [c for c in predictor.columns if c in OUTCOME_SOURCE_COLUMNS]
    assert not leaked, f"internet-use fields must not reach the modeling table: {leaked}"

    if write:
        os.makedirs(DATA_DIR, exist_ok=True)
        filtered.to_csv(FILTERED_FILE, index=False)
        predictor.to_csv(PREDICTOR_FILE, index=False)
    return predictor


# ---------------------------------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------------------------------

def summarize_cohort(raw_counts, cohort):
    """Per-year and total counts at each cohort step. raw_counts: {year: {step: n}}; cohort: outcome table."""
    rows = []
    for year in sorted(raw_counts):
        c = cohort[cohort["HRYEAR4"] == year]
        resolved = c[c["OUTCOME_STATUS"] == "resolved"]
        row = {"year": year, **raw_counts[year]}
        row.update({
            "cohort_six_categories": int(len(c)),
            "unresolved_dropped": int((c["OUTCOME_STATUS"] == "unresolved").sum()),
            "modeling_n": int(len(resolved)),
            "excluded_n": int(resolved[OUTCOME].sum()),
            "exclusion_rate": round(float(resolved[OUTCOME].mean()), 4) if len(resolved) else np.nan,
            "households": int(resolved.groupby(["HRHHID", "HRHHID2"]).ngroups) if len(resolved) else 0,
            "excluded_in_connected_household": int(((resolved["HEINHOME"] == 1) & (resolved[OUTCOME] == 1)).sum())
            if "HEINHOME" in resolved.columns else np.nan,
            "users_without_home_internet": int(((resolved["HEINHOME"] == 2) & (resolved[OUTCOME] == 0)).sum())
            if "HEINHOME" in resolved.columns else np.nan,
        })
        if "PWSSWGT" in resolved.columns and len(resolved):
            w = resolved["PWSSWGT"]
            row["weighted_exclusion_rate"] = round(float((w * resolved[OUTCOME]).sum() / w.sum()), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    total = summary.drop(columns=["year"]).sum(numeric_only=True).astype(object)
    for col in total.index:  # keep counts as integers in the total row
        if col not in ("exclusion_rate", "weighted_exclusion_rate"):
            total[col] = int(total[col])
    total["year"] = "total"
    for col in ["exclusion_rate", "weighted_exclusion_rate"]:
        if col in summary.columns:
            total[col] = np.nan  # rates do not add; recomputed below
    resolved_all = cohort[cohort["OUTCOME_STATUS"] == "resolved"]
    total["exclusion_rate"] = round(float(resolved_all[OUTCOME].mean()), 4) if len(resolved_all) else np.nan
    if "PWSSWGT" in resolved_all.columns and len(resolved_all):
        w = resolved_all["PWSSWGT"]
        total["weighted_exclusion_rate"] = round(float((w * resolved_all[OUTCOME]).sum() / w.sum()), 4)
    summary = pd.concat([summary, total.to_frame().T], ignore_index=True)
    return summary


# ---------------------------------------------------------------------------------------------------

def main(raw_dir=DATA_DIR, verbose=True):
    csv_files = find_raw_files(raw_dir)
    if not csv_files:
        print(f"No raw census files (CPS20*.csv or nov*pub.csv) were found in {raw_dir}/.")
        return None

    raw_counts = {}
    cohort_parts = []
    for f in csv_files:
        df = load_raw(f)
        year = int(df["HRYEAR4"].iloc[0])
        age_ok = df[df["PRTAGE"] >= MIN_AGE]
        raw_counts[year] = {
            "raw_rows": int(len(df)),
            "age_65_plus": int(len(age_ok)),
            "age_65_plus_prdisflg": int((age_ok["PRDISFLG"] == 1).sum()) if "PRDISFLG" in df.columns else np.nan,
        }
        cohort_parts.append(apply_cohort(df))
        if verbose:
            print(f"{os.path.basename(f)}: {len(df):,} rows -> cohort {len(cohort_parts[-1]):,}")

    cohort = pd.concat(cohort_parts, ignore_index=True)

    # RESULTANT OF PHASE 2a: outcome built while every source field is still present
    cohort = build_outcome(cohort)
    checks = audit_outcome(cohort)

    # RESULTANT OF PHASE 2b: modeling table with the ten predictor groups and the outcome
    predictor = feature_selection(cohort)

    os.makedirs(DATA_DIR, exist_ok=True)
    summary = summarize_cohort(raw_counts, cohort)
    summary.to_csv(SUMMARY_FILE, index=False)

    if verbose:
        print("\nOutcome audit:", checks)
        print("\nCohort summary:")
        print(summary.to_string(index=False))
        print(f"\nWrote {FILTERED_FILE}, {PREDICTOR_FILE}, {SUMMARY_FILE}")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the digital exclusion modeling table.")
    parser.add_argument("--raw-dir", default=DATA_DIR,
                        help="folder holding the raw CPS files (default: data/). Outputs always go to data/.")
    args = parser.parse_args()
    main(args.raw_dir)