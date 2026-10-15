# COMP5630-Project
This is a project for our COMP 5630 class titled "Machine Learning." This repo holds the code files used to perform this experiment


## Data
The raw census files are too large for GitHub, so they are not in this repo. Download the
November 2019, 2021, and 2023 CPS Computer and Internet Use Supplement public-use CSVs and place
them in a `data/` folder as `nov19pub.csv`, `nov21pub.csv`, and `nov23pub.csv`.

## Running
```
python dataPrep.py                    # builds data/combined_data.csv and data/predictor_data.csv
python -m unittest test_dataPrep -v   # runs the unit tests
```
