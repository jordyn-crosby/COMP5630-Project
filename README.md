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

## Running dataPrep.py
```
python dataPrep.py                    # builds data/combined_data.csv, data/filtered_data.csv and data/predictor_data.csv
python -m unittest test_dataPrep -v   # runs the unit tests
```

## Running main.py
```
python main.py                        # trains/tests models for each disability type (runs dataPrep first if data/predictor_data.csv is missing)
python -m unittest test_main -v       # runs the unit tests
```
