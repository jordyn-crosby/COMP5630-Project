# Main Python File: main.py
# The purpose of this file is to run all phases of the experiment
# including data preparation, model training, and model testing. It predicts digital exclusion
# (no internet use at any location) among adults 65+ with a disability, compares four models,
# checks performance across survey years, ranks feature importance, tests a reduced feature set, and tests an
# extended feature set with five extra household/geography predictors.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

from xgboost import XGBClassifier

import dataPrep
import plot_results

PREDICTOR_FILE = "data/predictor_data.csv"

# nominal census codes (the number itself has no order, e.g. state FIPS) -> one-hot encoded
CATEGORICAL_FEATURES = ["PESEX", "PTDTRACE", "PEMARITL", "HETENURE", "GESTFIPS", "PEMLR"]

# age and ordered codes (income bracket, education level) -> scaled
NUMERIC_FEATURES = ["PRTAGE", "HEFAMINC", "PEEDUCA"]

# disability type as six 0/1 indicators (a person can have several types) -> passed through unchanged
DISABILITY_FEATURES = ["DIS_HEARING", "DIS_VISION", "DIS_COGNITIVE", "DIS_PHYSICAL", "DIS_SELFCARE", "DIS_INDLIVING"]

# the ten predictors from the proposal: nine demographic/socioeconomic variables plus disability type
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DISABILITY_FEATURES

# extra non-behavioral predictors, used only in the separate "Extended" experiment
EXTENDED_NUMERIC_FEATURES = ["HRNUMHOU", "PRNMCHLD"]  # household size, own children under 18
EXTENDED_CATEGORICAL_FEATURES = ["GTMETSTA", "PEAFEVER", "PRCITSHP"]  # metro status, veteran status, citizenship
EXTENDED_FEATURES = EXTENDED_NUMERIC_FEATURES + EXTENDED_CATEGORICAL_FEATURES

# binary digital exclusion outcome (1 = does not use the internet at any location), built in dataPrep.py
TARGET = dataPrep.TARGET_COLUMN

# survey year column, used to train on earlier years and test on the latest one
YEAR_COLUMN = "HRYEAR4"
TEMPORAL_TRAIN_YEARS = [2019, 2021]
TEMPORAL_TEST_YEAR = 2023

# the six disability indicators are reported (and kept or dropped) together as one predictor
DISABILITY_GROUP = "DISABILITY_TYPE"

# number of top predictors kept in the reduced (parsimonious) model
REDUCED_FEATURE_COUNT = 4

# performance table and charts written by main()
RESULTS_FILE = "results/model_performance.csv"
BASELINE_NAME = "Baseline (always connected)"

# experiment labels used in the results table and charts
RANDOM_EXPERIMENT = "Random split"
TEMPORAL_EXPERIMENT = f"Train {TEMPORAL_TRAIN_YEARS[0]}+{TEMPORAL_TRAIN_YEARS[1]}, test {TEMPORAL_TEST_YEAR}"
REDUCED_EXPERIMENT = f"Reduced ({REDUCED_FEATURE_COUNT} predictors)"
ACCURACY_EXPERIMENT = "Accuracy-tuned (unweighted)"
EXTENDED_EXPERIMENT = f"Extended (+{len(EXTENDED_FEATURES)} predictors)"

# Hypothesis 1: test accuracy each model is expected to reach with the ten predictors
HYPOTHESIS_1_TARGETS = {
    "Logistic Regression": 0.783,
    "Decision Tree": 0.807,
    "Random Forest": 0.800,
    "XGBoost": 0.790,
}

# metric used both to tune hyperparameters and to pick the best model
# (macro F1 weighs both classes equally, so predicting only the majority class scores poorly)
SCORING = "f1_macro"


# Preprocessing: fit on training folds only because it lives inside each model's Pipeline
def build_preprocessor(features=FEATURES):
    return ColumnTransformer([
        ("num", StandardScaler(), [f for f in NUMERIC_FEATURES + EXTENDED_NUMERIC_FEATURES if f in features]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         [f for f in CATEGORICAL_FEATURES + EXTENDED_CATEGORICAL_FEATURES if f in features]),
        ("dis", "passthrough", [f for f in DISABILITY_FEATURES if f in features]),
    ])


# Each model with a small hyperparameter grid. balanced=True weights the classes equally (class-imbalance
# handling); balanced=False leaves them unweighted, which favors overall accuracy over catching excluded people
def build_models(y_train, balanced=True):
    # XGBoost has no class_weight option; weight positives by the negative/positive ratio instead
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1) if balanced else 1
    class_weight = "balanced" if balanced else None

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, solver="liblinear", class_weight=class_weight),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, class_weight=class_weight),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [1, 20, 50]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 20]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, n_jobs=1, scale_pos_weight=pos_weight),
            {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1], "model__n_estimators": [100, 300]},
        ),
    }


# Tune each model with cross-validation on the training data only
def fit_models(x_train, y_train, cv=5, balanced=True, scoring=SCORING):
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    features = list(x_train.columns)

    models = {}
    for name, (estimator, param_grid) in build_models(y_train, balanced=balanced).items():
        pipeline = Pipeline([("prep", build_preprocessor(features)), ("model", estimator)])
        search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=folds, n_jobs=-1)
        search.fit(x_train, y_train)

        models[name] = search.best_estimator_
        train_acc = accuracy_score(y_train, search.best_estimator_.predict(x_train))
        print(f"{name}: best params {search.best_params_}, CV {scoring}={search.best_score_:.3f}, "
              f"train accuracy={train_acc:.3f}")

    print(" Model Training Complete")
    return models


# Random 70/30 split, stratified so train and test keep the same class ratio
def train_models(df=None, target=TARGET, cv=5, features=FEATURES, balanced=True, scoring=SCORING):

    # read in data
    if df is None:
        df = pd.read_csv(PREDICTOR_FILE)

    x = df[features]
    y = df[target]  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    models = fit_models(x_train, y_train, cv=cv, balanced=balanced, scoring=scoring)
    return models, x_train, x_test, y_train, y_test


# Temporal check: train on earlier survey years, test on the latest one
def train_models_temporal(df=None, target=TARGET, cv=5, features=FEATURES):
    if df is None:
        df = pd.read_csv(PREDICTOR_FILE)

    train = df[df[YEAR_COLUMN].isin(TEMPORAL_TRAIN_YEARS)]
    test = df[df[YEAR_COLUMN] == TEMPORAL_TEST_YEAR]

    x_train, y_train = train[features], train[target]
    x_test, y_test = test[features], test[target]
    models = fit_models(x_train, y_train, cv=cv)
    return models, x_train, x_test, y_train, y_test


# model input columns that make up each predictor (the six disability indicators count as one predictor)
def predictor_groups(features):
    groups = {}
    for f in features:
        groups.setdefault(DISABILITY_GROUP if f in DISABILITY_FEATURES else f, []).append(f)
    return groups


# Permutation importance per predictor: how much the CV score (SCORING) drops when that predictor's columns are
# shuffled. Measured on held-out folds of the training data only, so the test set plays no part in choosing the
# reduced model. Each predictor is shuffled as one unit (raw columns, before one-hot encoding), so predictors with
# many categories (e.g. state) are not inflated. Negative drops count as 0; each model's scores sum to 1.
def feature_importance(models, x_train, y_train, cv=5, n_repeats=5):
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    groups = predictor_groups(x_train.columns)
    scorer = get_scorer(SCORING)
    rng = np.random.default_rng(42)

    table = {}
    for name, pipeline in models.items():
        drops = pd.Series(0.0, index=list(groups))
        for train_idx, val_idx in folds.split(x_train, y_train):
            model = clone(pipeline).fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            x_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]
            base = scorer(model, x_val, y_val)
            for group, columns in groups.items():
                for _ in range(n_repeats):
                    shuffled = x_val.copy()
                    shuffled[columns] = x_val[columns].to_numpy()[rng.permutation(len(x_val))]
                    drops[group] += base - scorer(model, shuffled, y_val)
        scores = (drops / (cv * n_repeats)).clip(lower=0)
        table[name] = scores / scores.sum() if scores.sum() > 0 else scores

    importance = pd.DataFrame(table)
    importance["Mean"] = importance.mean(axis=1)
    return importance.sort_values("Mean", ascending=False)


# expand predictor names (DISABILITY_TYPE -> the six indicators) into model input columns
def expand_features(predictors):
    columns = []
    for p in predictors:
        columns += DISABILITY_FEATURES if p == DISABILITY_GROUP else [p]
    return columns


# score one set of predictions against the real answers in the test set
def compute_metrics(name, y_test, y_pred, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro"),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, probabilities),
        "Confusion Matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "Classification Report": classification_report(y_test, y_pred, zero_division=0),
    }


# confusion matrix in plain words: rows are the real answer, columns are the model's prediction
def format_confusion_matrix(cm):
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    return "\n".join([
        f"Predicted vs actual (test set, n={total}):",
        f"{'':22}{'Predicted connected':>22}{'Predicted excluded':>24}",
        f"{'Actually connected':22}{f'{tn} (correct)':>22}{f'{fp} (false alarm)':>24}",
        f"{'Actually excluded':22}{f'{fn} (missed)':>22}{f'{tp} (correct)':>24}",
        f"Overall accuracy: {tn + tp} / {total} = {(tn + tp) / total:.3f}",
    ])


def test_models(models, x_test, y_test, select_by="Macro F1"):
    results = []

    for name, model in models.items():
        y_pred = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]  # Probability of the positive class
        results.append(compute_metrics(name, y_test, y_pred, probabilities))

    print(" Model Testing Complete. Initiating Model Comparison...\n\n")

    # Majority-class baseline: a useful model has to beat this
    majority = int(y_test.mode()[0])
    baseline_pred = pd.Series(majority, index=y_test.index)
    baseline = compute_metrics(BASELINE_NAME, y_test, baseline_pred, baseline_pred.astype(float))

    # Compare models by select_by, macro F1 by default (the baseline is reported but never picked as best)
    best_model = max(results, key=lambda r: r[select_by])

    print("Model Comparison:")
    for r in [baseline] + results:
        print(f"{r['Model']}: Accuracy={r['Accuracy']:.3f}, Balanced Acc={r['Balanced Accuracy']:.3f}, "
              f"Macro F1={r['Macro F1']:.3f}, AUC={r['ROC AUC']:.3f}")

    print(f"\nBest model: {best_model['Model']} with {select_by}={best_model[select_by]:.3f}")
    print(best_model["Classification Report"])
    print(format_confusion_matrix(best_model["Confusion Matrix"]))

    results.append(baseline)
    return results, best_model


# one CSV-ready row per model (and the baseline) for one experiment
def results_table(results, experiment):
    rows = []
    for r in results:
        (tn, fp), (fn, tp) = r["Confusion Matrix"]
        rows.append({
            "Experiment": experiment,
            "Model": r["Model"],
            "Test Size": int(tn + fp + fn + tp),
            "Accuracy": r["Accuracy"],
            "Balanced Accuracy": r["Balanced Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1 (excluded)": r["F1 Score"],
            "Macro F1": r["Macro F1"],
            "ROC AUC": r["ROC AUC"],
            "Correct Connected (TN)": int(tn),
            "False Alarms (FP)": int(fp),
            "Missed Excluded (FN)": int(fn),
            "Correct Excluded (TP)": int(tp),
        })
    return pd.DataFrame(rows).round(3)


def save_results(df, path=RESULTS_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved model performance table to {path}")


# Hypothesis 1: does each model's test accuracy reach its target?
def hypothesis_1_check(results):
    rows = [{
        "Model": r["Model"],
        "Accuracy": round(r["Accuracy"], 3),
        "Target": HYPOTHESIS_1_TARGETS[r["Model"]],
        "Gap to Target": round(r["Accuracy"] - HYPOTHESIS_1_TARGETS[r["Model"]], 3),
        "Met Target": r["Accuracy"] >= HYPOTHESIS_1_TARGETS[r["Model"]],
    } for r in results if r["Model"] in HYPOTHESIS_1_TARGETS]
    return pd.DataFrame(rows)


def summarize(label, results, best_model):
    return {
        "Experiment": label,
        "Best Model": best_model["Model"],
        "Accuracy": round(best_model["Accuracy"], 3),
        "Macro F1": round(best_model["Macro F1"], 3),
        "Balanced Accuracy": round(best_model["Balanced Accuracy"], 3),
        "Recall": round(best_model["Recall"], 3),
        "ROC AUC": round(best_model["ROC AUC"], 3),
    }


def main():

    # Step 1: Data Preparation (skipped when the predictor file already exists; delete it to rebuild)
    if not Path(PREDICTOR_FILE).exists():
        if dataPrep.main() is None:
            return None

    df = pd.read_csv(PREDICTOR_FILE)
    print(f"Population: {len(df)} adults 65+ with a disability; "
          f"digitally excluded rate = {df[TARGET].mean():.3f}")
    summary = []
    tables = []

    # Step 2 + 3: train and test on a random stratified 70/30 split
    print("\n===== Experiment: random 70/30 split, all ten predictors =====")
    models, x_train, x_test, y_train, y_test = train_models(df)
    results, best_model = test_models(models, x_test, y_test)
    summary.append(summarize(RANDOM_EXPERIMENT, results, best_model))
    tables.append(results_table(results, RANDOM_EXPERIMENT))

    # Step 3b: same split, but unweighted and tuned for accuracy, the measure Hypothesis 1 is stated in
    print("\n===== Experiment: random 70/30 split, unweighted classes, tuned for accuracy =====")
    a_models, _, a_x_test, _, a_y_test = train_models(df, balanced=False, scoring="accuracy")
    a_results, a_best = test_models(a_models, a_x_test, a_y_test, select_by="Accuracy")
    summary.append(summarize(ACCURACY_EXPERIMENT, a_results, a_best))
    tables.append(results_table(a_results, ACCURACY_EXPERIMENT))

    print("\n===== Hypothesis 1: test accuracy vs target =====")
    for label, res in [(RANDOM_EXPERIMENT, results), (ACCURACY_EXPERIMENT, a_results)]:
        print(f"\n{label}:")
        print(hypothesis_1_check(res).to_string(index=False))

    # Step 4: temporal check, train on 2019 + 2021 and test on 2023
    if YEAR_COLUMN in df.columns:
        print(f"\n===== Experiment: train on {TEMPORAL_TRAIN_YEARS}, test on {TEMPORAL_TEST_YEAR} =====")
        t_models, _, t_x_test, _, t_y_test = train_models_temporal(df)
        t_results, t_best = test_models(t_models, t_x_test, t_y_test)
        summary.append(summarize(TEMPORAL_EXPERIMENT, t_results, t_best))
        tables.append(results_table(t_results, TEMPORAL_EXPERIMENT))

    # Step 5: which predictors matter most
    importance = feature_importance(models, x_train, y_train)
    print("\n===== Permutation importance (CV on training data, normalized, grouped by predictor) =====")
    print(importance.round(3).to_string())

    # Step 6: parsimonious model on the top predictors only
    top = list(importance.index[:REDUCED_FEATURE_COUNT])
    print(f"\n===== Experiment: reduced model with top {REDUCED_FEATURE_COUNT} predictors {top} =====")
    r_models, _, r_x_test, _, r_y_test = train_models(df, features=expand_features(top))
    r_results, r_best = test_models(r_models, r_x_test, r_y_test)
    summary.append(summarize(REDUCED_EXPERIMENT, r_results, r_best))
    tables.append(results_table(r_results, REDUCED_EXPERIMENT))

    # Step 6b: extended model, the ten predictors plus five household/geography predictors
    # (same seed and stratified split as the random experiment, so both are tested on the same people)
    print(f"\n===== Experiment: extended model, ten predictors + {EXTENDED_FEATURES} =====")
    e_models, _, e_x_test, _, e_y_test = train_models(df, features=FEATURES + EXTENDED_FEATURES)
    e_results, e_best = test_models(e_models, e_x_test, e_y_test)
    summary.append(summarize(EXTENDED_EXPERIMENT, e_results, e_best))
    tables.append(results_table(e_results, EXTENDED_EXPERIMENT))
    print(f"\nHypothesis 1, {EXTENDED_EXPERIMENT}:")
    print(hypothesis_1_check(e_results).to_string(index=False))

    summary = pd.DataFrame(summary)
    print("\n===== Summary: best model per experiment =====")
    print(summary.to_string(index=False))

    # Step 7: save the performance table and draw the charts from it
    save_results(pd.concat(tables, ignore_index=True))
    plot_results.main(RESULTS_FILE)
    return summary


if __name__ == "__main__":
    main()
