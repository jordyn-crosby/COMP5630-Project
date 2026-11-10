# COMP5630-Project
This is a project for our COMP 5630 class titled "Machine Learning." This repo holds the code files used to perform this experiment


## Data
The raw census files are too large for GitHub, so they are not in this repo. Download the
November 2019, 2021, and 2023 CPS Computer and Internet Use Supplement public-use CSVs and place
them in a `data/` folder as `nov19pub.csv`, `nov21pub.csv`, and `nov23pub.csv`.

## Predictor Variable Key 
**PRTAGE** : AGE

**PESEX** : SEX

**PTDTRACE** : RACE

**PEMARITL** : MARITAL STATUS

**HEFAMINC** : FAMILY INCOME

**PEEDUCA** : EDUCATION

**HETENURE** : TENURE

**GESTFIPS** : STATE

**PEMLR** : LABOR FORCE PARTICIPATION

**DISTYPE** : DISABILITY TYPE

## Setup
```
python3 -m venv .venv
source .venv/bin/activate             # run this in every new terminal before the commands below
pip install -r docs/requirements.txt
```

## Project layout
```
src/      dataPrep.py, main.py, plot_results.py
tests/    unit tests for each script
data/     raw census CSVs and the files dataPrep.py builds from them (not in git)
results/  model performance table and charts written by main.py
docs/     requirements.txt and output.txt (console output from a full run)
```
The scripts locate `data/` and `results/` from their own location, so they can be run from the
project root or from inside `src/`. Run the test commands below from the project root.

## Running the tests
```
python -m unittest -v                     # runs every test in tests/
```

## Running dataPrep.py
```
python src/dataPrep.py                    # builds data/combined_data.csv, data/filtered_data.csv and data/predictor_data.csv
python -m unittest tests.test_dataPrep -v # runs the unit tests
```

## Running main.py
```
python src/main.py                        # predicts digital exclusion: random split, 2019+2021 -> 2023 check, feature importance, reduced model
python -m unittest tests.test_main -v     # runs the unit tests
```
`main.py` writes `results/model_performance.csv` (accuracy, macro F1, ROC AUC and predicted-vs-actual
counts for every model in every experiment) and draws `results/model_performance.png` and
`results/confusion_matrix.png` from it.

## Redrawing the charts
```
python src/plot_results.py                # rebuilds the charts from results/model_performance.csv, no retraining
python -m unittest tests.test_plot_results -v
```
