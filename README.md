# Who Gets Left Behind? Machine Learning for Digital Exclusion Among Older Adults with Disabilities

COMP 6630 Machine Learning, Auburn University, Fall 2026. Jordyn Crosby and Meena Hall.

Predicts digital exclusion (no Internet use at any location) among U.S. adults aged 65+ with a disability
from the November 2019, 2021 and 2023 CPS Computer and Internet Use Supplements, using ten groups of
intake-style characteristics and four classifiers (logistic regression, decision tree, random forest, XGBoost).

## One-command run

```
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r docs/requirements.txt
python src/run_all.py
```

`run_all.py` extracts the raw files from `data/data.zip` if they are not already in `data/`, builds the modeling
table, runs the 58 unit tests, runs Experiment B (temporal) then Experiment A (pooled), the H3 feature-reduction
check and the sensitivity runs C and D, and draws the figures. About 15 minutes on a laptop. Add `--raw-dir /folder`
if the CPS files live elsewhere, `--skip-tests` or `--skip-sensitivity` to shorten it, `--quick` for a smoke test.

The committed `data/predictor_data.csv` (18,762 rows) lets `python src/main.py` run immediately without the
raw files; `src/dataPrep.py` rebuilds it from the raw files in about 5 seconds.

## Layout

| Path | Contents |
|---|---|
| `src/dataPrep.py` | reads only the ~45 needed columns from each raw file, harmonizes GCFIP/GESTFIPS, applies the cohort rule (age 65+ and any of the six CPS disability categories), builds the outcome before dropping any source field, writes `data/filtered_data.csv` (audit), `data/predictor_data.csv` (modeling), `data/cohort_summary.csv` |
| `src/main.py` | Experiment B (tune on 2019+2021 with household-grouped stratified CV, evaluate once on 2023), Experiment A (pooled, outcome-stratified 70/30 split keeping households together), optional sensitivity runs C (`--experiment C`: unweighted classes, tuned on accuracy) and D (`--experiment D`: A's split with five extra predictors: household size, children under 18, metropolitan status, veteran status, citizenship) |
| `src/h3_feature_reduction.py` | full ten-group model vs the prespecified reduced set (age, income, education, disability type) with tolerances fixed before results are seen |
| `src/plot_results.py` | `results/model_performance.csv` and the two figures, drawn from the saved outputs |
| `src/run_all.py` | the one-command pipeline above |
| `tests/` | 55 unit tests: cohort rule, the five outcome validation cases, no internet fields in inputs, disjoint households, saved files, figures |
| `docs/outcome_specification.md` | the outcome dictionary: source fields, codes, skip logic, coding rule, validation cases, cohort counts |
| `docs/requirements.txt` | Python dependencies |
| `data/data.zip` | the three raw CPS public-use files (nov19pub.csv, nov21pub.csv, nov23pub.csv), 44 MB compressed |
| `outputs/<experiment>/` | per experiment: `model_comparison.csv`, `test_predictions.csv`, `best_params.json`, `cv_summary.csv`, `cv_folds.csv`, `confusion_matrices.csv`, `classification_reports.txt`, `hypotheses.csv`, `run_manifest.json`, and for A and B `logistic_coefficients.csv`, `tree_importances.csv`, `permutation_importance.csv` |
| `results/` | `model_performance.csv`, `model_performance.png`, `confusion_matrix.png`, `COMP6630_Results_Six_Categories.xlsx` |

## Variable key

| Group | Column | Notes |
|---|---|---|
| Age | PRTAGE | top-coded 80 (80 to 84) and 85 (85+); standardized |
| Sex | PESEX | one-hot |
| Race | PTDTRACE | one-hot |
| Education | PEEDUCA | ordered code; standardized |
| Family income | HEFAMINC | 16 ordered brackets; standardized (Census-allocated when missing, flag HXFAMINC in the audit table) |
| Housing tenure | HETENURE | one-hot |
| Marital status | PEMARITL | one-hot |
| Employment status | PEMLR | one-hot |
| State | GESTFIPS (GCFIP in 2019) | one-hot, 51 codes |
| Disability type | DIS_HEARING, DIS_VISION, DIS_COGNITIVE, DIS_MOBILITY, DIS_SELFCARE, DIS_INDLIVING | 0/1 from PEDISEAR, PEDISEYE, PEDISREM, PEDISPHY, PEDISDRS, PEDISOUT |
| Outcome | DIGITAL_EXCLUSION | 1 = reported as using the Internet at no location (after the household follow-up), 0 = user |
| Metadata | HRYEAR4, HRHHID, HRHHID2, PULINENO, HRNUMHOU, PWSSWGT | grouping, ids, survey weight; never inputs |

## Evaluation protocol

* Cohort: 18,762 people (6,610 / 5,884 / 6,268 by year), 16,875 households, 7,208 excluded (38.4%); exclusion
  falls from 44.9% in 2019 to 32.4% in 2023.
* Preprocessing lives inside each model's pipeline, so it is fit on training folds only; unseen categories are
  ignored, not errors. All four models are tuned with `GridSearchCV` on identical household-grouped,
  outcome-stratified folds, scored on macro F1, with class weighting. A majority-class dummy fitted on training
  labels is the baseline.
* Experiment B runs before A with everything frozen; nothing from 2023 or from the pooled run feeds back into
  tuning. Runs C and D are labeled sensitivity analyses (`protocol_experiment: false` in their manifests), never
  headlines.
* Headline results (held-out): pooled accuracy 67.1 to 68.4% and ROC AUC 0.69 to 0.74 against 61.6% / 0.50 for
  the baseline; 2023 accuracy 66.0 to 67.6%, ROC AUC 0.70 to 0.73. H1 (literature accuracy targets) not met;
  H2 (ensembles above single models) holds against the tree but not against logistic regression; H3 (reduced
  set within tolerance) holds for all four models. Details in `outputs/*/hypotheses.csv`.
