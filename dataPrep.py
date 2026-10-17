# Title: dataPrep.py
# Description: This Python script is design to prepare and preprocess the data for COMP5630 project.
# It completes phases 1 and 2 of the research project

# Phase 1: Data Preparation
# Collect 2019,2021,2023 raw census data
# Combine data into one large CSV file
# Filter data based on age and disability flag

# Phase 2: Feature Selection
# Select predictor variables and target variable (disability type)
# Exclude direct internet access and technology ownership characteristics

import glob
import pandas as pd

# columns whose names differ between survey years, mapped to a common name
# (2019 stores state FIPS as GCFIP; 2021 and 2023 use GESTFIPS)
COLUMN_RENAMES = {
    "GCFIP": "GESTFIPS",
}

# concatenate multiple CSV files into a single DataFrame and save it to a new CSV file
def concat_csv_files(file_list, output_file):
    df_list = [pd.read_csv(file).rename(columns=COLUMN_RENAMES) for file in file_list]
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(output_file, index=False)

# load data from a CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# filtering out data based on a list of conditions
def filter_data(df, filter_list):
    mask = pd.Series(True, index=df.index)
    for condition in filter_list:
        mask &= condition
    return df[mask].copy()


def feature_selection(df):
    # census flag column -> (disability name, 0/1 indicator column)
    disability_map = {
        "PEDISEAR": ("Hearing", "DIS_HEARING"),
        "PEDISEYE": ("Vision", "DIS_VISION"),
        "PEDISREM": ("Cognitive", "DIS_COGNITIVE"),
        "PEDISPHY": ("Physical", "DIS_PHYSICAL"),
        "PEDISDRS": ("Self-Care", "DIS_SELFCARE"),
        "PEDISOUT": ("Independent Living", "DIS_INDLIVING")
    }

    # one 0/1 column per disability type (1 = census flag is 1 = "yes");
    # a person can have several, so these are the multi-label targets
    indicator_columns = []
    for flag_col, (name, indicator_col) in disability_map.items():
        if flag_col in df.columns:
            df[indicator_col] = (df[flag_col] == 1).astype(int)
        else:
            df[indicator_col] = 0
        indicator_columns.append(indicator_col)

    # readable summary of every type a person has, e.g. "Hearing|Physical"
    # (split back into indicators with df["DISTYPE"].str.get_dummies("|"))
    names = pd.Series([name + "|" for name, _ in disability_map.values()], index=indicator_columns)
    df["DISTYPE"] = df[indicator_columns].astype(bool).dot(names).str.rstrip("|").replace("", "None")
    # filtered rows with every column kept (separate file so the raw combined_data.csv is not overwritten)
    df.to_csv("data/filtered_data.csv", index=False)

    predictor_variables = [
        "PRTAGE",    # age
        "PESEX",     # sex
        "PTDTRACE",  # race
        "PEMARITL",  # marital status
        "HEFAMINC",  # family income
        "PEEDUCA",   # education
        "HETENURE",  # housing tenure
        "GESTFIPS",  # state
        "PEMLR",     # employment status
        "DISTYPE"
    ] + indicator_columns

    columns_to_drop = [col for col in df.columns if col not in predictor_variables]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df.to_csv("data/predictor_data.csv", index=False)
    return df


def main():

    # only the raw survey files; data/*.csv would also pick up this script's own outputs
    csv_files = sorted(glob.glob("data/nov*pub.csv"))

    if not csv_files:
        print("No raw census files (nov*pub.csv) were found in the data/ folder.")
        return None

    combined_file = "data/combined_data.csv"
    concat_csv_files(csv_files, combined_file)

    df = load_data(combined_file)
    print(f"Combined {len(csv_files)} CSV files into {combined_file}")

    # RESULTANT OF PHASE 1: filtering based on age and disability flag
    df = filter_data(df, [df['PRTAGE'] >= 65,df['PRDISFLG'] == 1 ])

    # RESULTANT OF PHASE 2: feature selection based on predictor variables and target variable (disability type)
    df = feature_selection(df)

    return df


if __name__ == "__main__":
    main()