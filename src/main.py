# Main Python File: main.py
# Runs the two promised experiments for "Who Gets Left Behind?" (COMP 6630, Hall & Crosby):
#
#   Experiment B (temporal, run first): develop and tune on 2019 + 2021, refit on all development
#                                        records, evaluate once on 2023.
#   Experiment A (pooled):               pool the three years, outcome-stratified 70/30 split that keeps
#                                        household members together, tune with k-fold CV inside the
#                                        training set, evaluate on the held-out 30 percent.
#
# One binary outcome (DIGITAL_EXCLUSION), fifteen raw inputs across the ten predictor groups, a
# majority-class dummy baseline fitted on training labels, and four classifiers (logistic regression,
# decision tree, random forest, XGBoost) tuned on identical CV folds with the same scoring rule.
# All preprocessing lives inside each model's Pipeline, so it is fit on training folds only.
# Model families, grids, outcome rule and scoring are frozen in this file before either experiment runs;
# nothing from the test sets feeds back into tuning.
#
# Saved per experiment under outputs/<experiment>/:
#   model_comparison.csv, test_predictions.csv, best_params.json, cv_summary.csv, cv_folds.csv,
#   confusion_matrices.csv, classification_reports.txt, logistic_coefficients.csv,
#   tree_importances.csv, permutation_importance.csv, hypotheses.csv, run_manifest.json
#
# Run:  python main.py                 (both experiments; runs dataPrep first if data/predictor_data.csv is missing)
#       python main.py --experiment A  (pooled only)   python main.py --experiment B  (temporal only)
#       python main.py --cv 3 --quick  (smaller grids, for a smoke test)

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import dataPrep

PREDICTOR_FILE = dataPrep.PREDICTOR_FILE
OUTPUT_DIR = "outputs"
TARGET = dataPrep.OUTCOME  # DIGITAL_EXCLUSION: 1 = excluded (uses the Internet nowhere), 0 = user
SEED = 42

# ----------------------------------------------------------------------------------------------------
# Inputs: ten predictor groups -> fifteen raw columns
# ----------------------------------------------------------------------------------------------------
# age and ordered codes (income bracket, education level) -> standardized
NUMERIC_FEATURES = ["PRTAGE", "HEFAMINC", "PEEDUCA"]
# nominal census codes (the number itself has no order, e.g. state FIPS) -> one-hot encoded
CATEGORICAL_FEATURES = ["PESEX", "PTDTRACE", "PEMARITL", "HETENURE", "GESTFIPS", "PEMLR"]
# disability type: six 0/1 flags (overlap preserved) -> passed through unchanged
BINARY_FEATURES = list(dataPrep.DISABILITY_INDICATORS)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# conceptual group of every raw column (used to aggregate importances back to the ten groups)
FEATURE_GROUPS = {
    "PRTAGE": "Age", "PESEX": "Sex", "PTDTRACE": "Race", "PEEDUCA": "Education", "HEFAMINC": "Family income",
    "HETENURE": "Housing tenure", "PEMARITL": "Marital status", "PEMLR": "Employment status", "GESTFIPS": "State",
    **{col: "Disability type" for col in BINARY_FEATURES},
}

# H3 reduced set, prespecified in the proposal: education, household income, age, disability type
REDUCED_FEATURES = ["PRTAGE", "HEFAMINC", "PEEDUCA"] + BINARY_FEATURES

# metadata that identifies records and groups but is never an input
GROUP_COLUMNS = ["HRYEAR4", "HRHHID", "HRHHID2"]
ID_COLUMNS = GROUP_COLUMNS + ["PULINENO"]
INTERNET_COLUMNS = set(dataPrep.OUTCOME_SOURCE_COLUMNS) | {"INTERNET_USER"}

# metric used to tune hyperparameters (macro F1 weighs both classes equally)
SCORING = "f1_macro"
TEST_FRACTION = 0.30
SPLIT_FOLDS = 10  # 10 stratified group folds, 3 of them form the 30 percent test set

# H1 accuracy targets from the proposal (percent), one per model
H1_TARGETS = {"Logistic Regression": 78.3, "Decision Tree": 80.7, "Random Forest": 80.0, "XGBoost": 79.0}

MODEL_ORDER = ["Dummy (majority)", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


# ----------------------------------------------------------------------------------------------------
# Preprocessing and models
# ----------------------------------------------------------------------------------------------------

def build_preprocessor(features=None):
    """Fit on training folds only because it lives inside each model's Pipeline."""
    features = FEATURES if features is None else features
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    categorical = [f for f in CATEGORICAL_FEATURES if f in features]
    binary = [f for f in BINARY_FEATURES if f in features]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    if binary:
        transformers.append(("bin", "passthrough", binary))
    return ColumnTransformer(transformers)


def build_models(y_train, quick=False, class_weighting=True):
    """
    Each classifier with a compact hyperparameter grid. With class_weighting (the protocol), the minority
    class is up-weighted ("balanced", scale_pos_weight for XGBoost); without it (sensitivity experiment C)
    the models are fit on raw labels.
    """
    # XGBoost has no class_weight option; weight positives by the negative/positive ratio instead
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if class_weighting else 1.0
    cw = "balanced" if class_weighting else None

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", class_weight=cw),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED, class_weight=cw),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [1, 20, 50]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight=cw, n_jobs=1),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 20]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, n_jobs=1, scale_pos_weight=pos_weight, eval_metric="logloss"),
            {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1], "model__n_estimators": [100, 300]},
        ),
    }
    if quick:  # smoke-test grids
        models = {
            "Logistic Regression": (models["Logistic Regression"][0], {"model__C": [1]}),
            "Decision Tree": (models["Decision Tree"][0], {"model__max_depth": [5]}),
            "Random Forest": (models["Random Forest"][0].set_params(n_estimators=50), {"model__max_depth": [10]}),
            "XGBoost": (models["XGBoost"][0], {"model__max_depth": [3], "model__n_estimators": [100]}),
        }
    return models


# ----------------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------------

def make_group_key(df):
    """Household key. Year is included so identical household ids in different years never merge."""
    return df["HRYEAR4"].astype(str) + "_" + df["HRHHID"].astype(str) + "_" + df["HRHHID2"].astype(str)


def pooled_split(df, test_fraction=TEST_FRACTION, n_folds=SPLIT_FOLDS, seed=SEED):
    """
    Outcome-stratified train/test split that keeps household members on the same side.
    Builds n_folds stratified group folds and assigns round(test_fraction * n_folds) of them to the test
    set, so the realized test share is close to (but not exactly) the target. The realized proportions
    are reported in run_manifest.json.
    """
    groups = make_group_key(df)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(df, df[TARGET], groups))
    n_test_folds = int(round(test_fraction * n_folds))
    test_idx = np.sort(np.concatenate([folds[i][1] for i in range(n_test_folds)]))
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    return df[train_mask].copy(), df.iloc[test_idx].copy()


def temporal_split(df, dev_years=(2019, 2021), test_year=2023):
    dev = df[df["HRYEAR4"].isin(dev_years)].copy()
    test = df[df["HRYEAR4"] == test_year].copy()
    return dev, test


def cv_folds(train_df, k=5, seed=SEED):
    """Identical stratified, household-grouped folds for every model."""
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(train_df, train_df[TARGET], make_group_key(train_df)))


def check_disjoint_groups(train_df, test_df):
    overlap = set(make_group_key(train_df)) & set(make_group_key(test_df))
    assert not overlap, f"{len(overlap)} households appear in both training and test data"
    return True


# ----------------------------------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------------------------------

def train_models(train_df, folds, features=None, quick=False, verbose=True, scoring=SCORING,
                 class_weighting=True):
    """Tune each model with GridSearchCV on the shared folds; return fitted searches and a CV summary."""
    features = FEATURES if features is None else features
    x_train, y_train = train_df[features], train_df[TARGET]

    searches, cv_rows, fold_rows = {}, [], []
    for name, (estimator, grid) in build_models(y_train, quick, class_weighting).items():
        t0 = time.time()
        pipeline = Pipeline([("prep", build_preprocessor(features)), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, scoring=scoring, cv=folds, n_jobs=-1, refit=True,
                              return_train_score=True)
        search.fit(x_train, y_train)
        searches[name] = search

        res = search.cv_results_
        best = search.best_index_
        cv_rows.append({
            "Model": name, "Best Params": json.dumps(search.best_params_),
            f"CV {scoring} mean": round(float(res["mean_test_score"][best]), 4),
            f"CV {scoring} std": round(float(res["std_test_score"][best]), 4),
            f"CV train {scoring} mean": round(float(res["mean_train_score"][best]), 4),
            "Candidates": int(len(res["params"])), "Fit seconds": round(time.time() - t0, 1),
        })
        for i in range(len(folds)):
            fold_rows.append({"Model": name, "Fold": i, f"val {scoring}": float(res[f"split{i}_test_score"][best]),
                              f"train {scoring}": float(res[f"split{i}_train_score"][best])})
        if verbose:
            print(f"  {name}: best {search.best_params_}  CV {scoring}={search.best_score_:.3f}  "
                  f"({cv_rows[-1]['Fit seconds']}s)")

    # Majority-class dummy fitted on training labels only
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    if verbose:
        print("  Model training complete")
    return searches, dummy, pd.DataFrame(cv_rows), pd.DataFrame(fold_rows)


def metrics_for(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (excluded)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (excluded)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (excluded)": f1_score(y_true, y_pred, zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "ROC AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5,
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        "Confusion Matrix": cm,
        "Classification Report": classification_report(y_true, y_pred, target_names=["user", "excluded"],
                                                       zero_division=0),
    }


def test_models(searches, dummy, test_df, features=None, verbose=True):
    """Score every tuned model and the dummy on the same held-out records."""
    features = FEATURES if features is None else features
    x_test, y_test = test_df[features], test_df[TARGET]

    results = []
    predictions = test_df[ID_COLUMNS + [TARGET]].copy()
    y_dummy = dummy.predict(x_test)
    results.append(metrics_for("Dummy (majority)", y_test, y_dummy, dummy.predict_proba(x_test)[:, 1]))
    predictions["pred_Dummy"] = y_dummy
    for name, search in searches.items():
        model = search.best_estimator_
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        results.append(metrics_for(name, y_test, y_pred, y_prob))
        key = name.replace(" ", "_")
        predictions[f"pred_{key}"] = y_pred
        predictions[f"prob_{key}"] = np.round(y_prob, 5)

    if verbose:
        print("  Model testing complete\n")
        print("  Model comparison on held-out records:")
        for r in results:
            print(f"    {r['Model']:<20} Acc={r['Accuracy']:.3f}  BalAcc={r['Balanced Accuracy']:.3f}  "
                  f"Prec={r['Precision (excluded)']:.3f}  Rec={r['Recall (excluded)']:.3f}  "
                  f"F1={r['F1 (excluded)']:.3f}  MacroF1={r['Macro F1']:.3f}  AUC={r['ROC AUC']:.3f}")
    return results, predictions


# ----------------------------------------------------------------------------------------------------
# Interpretation
# ----------------------------------------------------------------------------------------------------

def logistic_coefficients(search, train_df, features=None):
    """
    Coefficients of the tuned logistic regression on the encoded design matrix.
    Numeric inputs are standardized, so 'coef' is the change in log odds of exclusion per one standard
    deviation; 'coef_per_unit' divides by the training SD to give the change per raw unit (per year of age,
    per income bracket, per education code). Categorical inputs are fully one-hot encoded (no dropped
    level) with L2 regularization, so each category coefficient is relative to the regularized average;
    'coef_vs_reference' re-expresses it against the most frequent training category of that variable.
    Disability flags are 0/1, so the coefficient is the log odds change for having that difficulty.
    """
    features = FEATURES if features is None else features
    pipe = search.best_estimator_
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    coefs = model.coef_.ravel()
    rows = []
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    scaler = prep.named_transformers_["num"] if numeric else None
    for n, c in zip(names, coefs):
        kind, raw = n.split("__", 1)
        if kind == "num":
            sd = float(scaler.scale_[numeric.index(raw)])
            rows.append({"input": raw, "category": "", "group": FEATURE_GROUPS[raw], "coef": c,
                         "coef_per_unit": c / sd if sd else np.nan, "coef_vs_reference": np.nan,
                         "unit": "per 1 SD (coef) / per raw unit (coef_per_unit)"})
        elif kind == "cat":
            var, cat = raw.rsplit("_", 1)
            rows.append({"input": var, "category": cat, "group": FEATURE_GROUPS[var], "coef": c,
                         "coef_per_unit": np.nan, "coef_vs_reference": np.nan, "unit": "category vs average"})
        else:
            rows.append({"input": raw, "category": "1 = has difficulty", "group": FEATURE_GROUPS[raw], "coef": c,
                         "coef_per_unit": c, "coef_vs_reference": np.nan, "unit": "0/1 flag"})
    table = pd.DataFrame(rows)
    # contrast each category against the most frequent training category of the same variable
    for var in [f for f in CATEGORICAL_FEATURES if f in features]:
        ref = str(train_df[var].mode().iloc[0])
        mask = table["input"] == var
        ref_row = table[mask & (table["category"] == ref)]
        if len(ref_row):
            table.loc[mask, "coef_vs_reference"] = table.loc[mask, "coef"] - float(ref_row["coef"].iloc[0])
            table.loc[mask, "unit"] = f"category vs reference {var}={ref}"
    table["odds_ratio"] = np.exp(table["coef"])
    table["intercept"] = float(model.intercept_[0])
    return table.sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)


def tree_importances(searches, features=None):
    """Impurity (DT, RF) and gain (XGB) importances aggregated from encoded columns back to raw inputs/groups."""
    features = FEATURES if features is None else features
    rows = []
    for name in ["Decision Tree", "Random Forest", "XGBoost"]:
        if name not in searches:
            continue
        pipe = searches[name].best_estimator_
        prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
        names = prep.get_feature_names_out()
        imp = np.asarray(model.feature_importances_, dtype=float)
        agg = {}
        for n, v in zip(names, imp):
            kind, raw = n.split("__", 1)
            var = raw.rsplit("_", 1)[0] if kind == "cat" else raw
            agg[var] = agg.get(var, 0.0) + float(v)
        for var, v in agg.items():
            rows.append({"Model": name, "input": var, "group": FEATURE_GROUPS.get(var, var), "importance": v})
    table = pd.DataFrame(rows)
    if len(table):
        table["group_importance"] = table.groupby(["Model", "group"])["importance"].transform("sum")
        table = table.sort_values(["Model", "importance"], ascending=[True, False]).reset_index(drop=True)
    return table


def permutation_importances(searches, test_df, features=None, n_repeats=10, seed=SEED):
    """
    Permutation importance on the held-out records, computed on the raw input columns so each conceptual
    input (e.g. state as one column) gets one score. Post-hoc interpretation only: nothing here is used
    to choose models or parameters. Importance is predictive contribution, not causal effect.
    """
    features = FEATURES if features is None else features
    x_test, y_test = test_df[features], test_df[TARGET]
    rows = []
    for name, search in searches.items():
        pi = permutation_importance(search.best_estimator_, x_test, y_test, scoring=SCORING,
                                    n_repeats=n_repeats, random_state=seed, n_jobs=-1)
        for f, m, s in zip(features, pi.importances_mean, pi.importances_std):
            rows.append({"Model": name, "input": f, "group": FEATURE_GROUPS[f],
                         f"mean drop in {SCORING}": float(m), "std": float(s)})
    table = pd.DataFrame(rows)
    return table.sort_values(["Model", f"mean drop in {SCORING}"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------
# Hypotheses
# ----------------------------------------------------------------------------------------------------

def evaluate_hypotheses(results):
    """H1 accuracy targets and H2 ensemble ordering, judged on the held-out results of one experiment."""
    by_name = {r["Model"]: r for r in results}
    rows = []
    for model, target in H1_TARGETS.items():
        if model in by_name:
            acc = 100 * by_name[model]["Accuracy"]
            rows.append({"Hypothesis": "H1", "Model": model, "Metric": "Accuracy (%)", "Value": round(acc, 2),
                         "Target": target, "Met": bool(acc >= target)})
    h1_any = any(r["Met"] for r in rows)
    rows.append({"Hypothesis": "H1", "Model": "any model", "Metric": "at least one target met", "Value": h1_any,
                 "Target": True, "Met": h1_any})
    for metric in ["Accuracy", "Macro F1", "ROC AUC"]:
        ens = [by_name[m][metric] for m in ["Random Forest", "XGBoost"] if m in by_name]
        single = [by_name[m][metric] for m in ["Logistic Regression", "Decision Tree"] if m in by_name]
        if ens and single:
            holds = min(ens) > max(single)
            rows.append({"Hypothesis": "H2", "Model": "RF & XGB vs LR & DT", "Metric": metric,
                         "Value": f"ensembles {min(ens):.4f}-{max(ens):.4f} vs singles {min(single):.4f}-{max(single):.4f}",
                         "Target": "both ensembles above both single models", "Met": bool(holds)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------------------------------

def save_results(outdir, results, predictions, cv_summary, cv_fold_table, searches, extras, manifest):
    os.makedirs(outdir, exist_ok=True)
    comparison = pd.DataFrame([{k: v for k, v in r.items() if k not in ("Confusion Matrix", "Classification Report")}
                               for r in results])
    comparison.to_csv(os.path.join(outdir, "model_comparison.csv"), index=False)
    predictions.to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)
    cv_summary.to_csv(os.path.join(outdir, "cv_summary.csv"), index=False)
    cv_fold_table.to_csv(os.path.join(outdir, "cv_folds.csv"), index=False)
    with open(os.path.join(outdir, "best_params.json"), "w") as f:
        json.dump({name: s.best_params_ for name, s in searches.items()}, f, indent=2)
    cm_rows = []
    for r in results:
        for true_label, pred_label, n in [("user", "user", r["TN"]), ("user", "excluded", r["FP"]),
                                          ("excluded", "user", r["FN"]), ("excluded", "excluded", r["TP"])]:
            cm_rows.append({"Model": r["Model"], "true": true_label, "predicted": pred_label, "count": n})
    pd.DataFrame(cm_rows).to_csv(os.path.join(outdir, "confusion_matrices.csv"), index=False)
    with open(os.path.join(outdir, "classification_reports.txt"), "w") as f:
        for r in results:
            f.write(f"===== {r['Model']} =====\n{r['Classification Report']}\n"
                    f"Confusion matrix [[TN FP] [FN TP]]:\n{r['Confusion Matrix']}\n\n")
    for key, table in extras.items():
        if table is not None and len(table):
            table.to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
    with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def base_manifest(name, description, features, cv, quick, scoring=SCORING, class_weighting=True):
    return {
        "experiment": name, "description": description,
        "protocol_experiment": bool(scoring == SCORING and class_weighting),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "seed": SEED, "scoring": scoring, "class_weighting": class_weighting, "cv_folds": cv, "quick_grids": quick,
        "target": TARGET, "features": list(features),
        "feature_groups": sorted(set(FEATURE_GROUPS[f] for f in features)),
        "internet_fields_in_inputs": sorted(set(features) & INTERNET_COLUMNS),
        "frozen_before_test_evaluation": ["model families", "hyperparameter grids", "outcome rule", "scoring rule",
                                          "cohort rule", "split procedure"],
    }


def describe(df):
    return {"n": int(len(df)), "excluded": int(df[TARGET].sum()), "exclusion_rate": round(float(df[TARGET].mean()), 4),
            "households": int(make_group_key(df).nunique()),
            "years": {int(k): int(v) for k, v in df["HRYEAR4"].value_counts().sort_index().items()}}


def run_experiment(name, train_df, test_df, outdir, description, features=None, cv=5, quick=False,
                   interpret=True, verbose=True, scoring=SCORING, class_weighting=True):
    features = FEATURES if features is None else features
    assert TARGET not in features and not (set(features) & INTERNET_COLUMNS), "outcome fields found in inputs"
    check_disjoint_groups(train_df, test_df)
    folds = cv_folds(train_df, k=cv)

    if verbose:
        print(f"\n===== Experiment {name}: {description} =====")
        print(f"  train/dev: {describe(train_df)}")
        print(f"  test:      {describe(test_df)}")

    searches, dummy, cv_summary, cv_fold_table = train_models(train_df, folds, features, quick, verbose,
                                                              scoring, class_weighting)
    results, predictions = test_models(searches, dummy, test_df, features, verbose)
    hypotheses = evaluate_hypotheses(results)

    extras = {"hypotheses": hypotheses}
    if interpret:
        extras["logistic_coefficients"] = logistic_coefficients(searches["Logistic Regression"], train_df, features)
        extras["tree_importances"] = tree_importances(searches, features)
        extras["permutation_importance"] = permutation_importances(searches, test_df, features)

    manifest = base_manifest(name, description, features, cv, quick, scoring, class_weighting)
    manifest.update({"train": describe(train_df), "test": describe(test_df),
                     "realized_test_fraction": round(len(test_df) / (len(test_df) + len(train_df)), 4),
                     "grids": {n: g for n, (_, g) in build_models(train_df[TARGET], quick, class_weighting).items()},
                     "best_params": {n: s.best_params_ for n, s in searches.items()}})
    save_results(outdir, results, predictions, cv_summary, cv_fold_table, searches, extras, manifest)
    if verbose:
        print(f"\n  Hypotheses:\n{hypotheses.to_string(index=False)}")
        print(f"  Saved results to {outdir}/")
    return results, searches


def load_predictor_table():
    if not os.path.exists(PREDICTOR_FILE):
        if dataPrep.main() is None:
            return None
    df = pd.read_csv(PREDICTOR_FILE)
    missing = [c for c in FEATURES + [TARGET] + ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{PREDICTOR_FILE} is missing {missing}; delete it and rerun dataPrep.py to rebuild")
    leaked = sorted(set(df.columns) & INTERNET_COLUMNS)
    if leaked:
        raise ValueError(f"{PREDICTOR_FILE} contains internet-use fields {leaked}; rebuild it with dataPrep.py")
    return df


def main(experiment="both", cv=5, quick=False, outdir=OUTPUT_DIR, verbose=True):
    df = load_predictor_table()
    if df is None:
        return None
    summary = {}

    # Experiment B first: the temporal evaluation is run before pooled results exist, so nothing from the
    # pooled comparison can influence the frozen temporal setup.
    if experiment in ("both", "B"):
        dev, test = temporal_split(df)
        results, _ = run_experiment(
            "B_temporal", dev, test, os.path.join(outdir, "B_temporal"),
            "develop and tune on 2019+2021 (household-grouped stratified CV), refit on all development records, "
            "evaluate once on 2023", cv=cv, quick=quick, verbose=verbose)
        summary["B_temporal"] = results

    if experiment in ("both", "A"):
        train, test = pooled_split(df)
        results, _ = run_experiment(
            "A_pooled", train, test, os.path.join(outdir, "A_pooled"),
            "pooled 2019+2021+2023, outcome-stratified ~70/30 split keeping households together, "
            "household-grouped stratified CV inside training", cv=cv, quick=quick, verbose=verbose)
        summary["A_pooled"] = results

    # Sensitivity experiment C (not part of the protocol, never a headline): the pooled split again, but with
    # no class weighting and tuning on plain accuracy. It shows what the models do when accuracy alone is
    # rewarded: accuracy rises a little and recall for the excluded class falls sharply.
    if experiment == "C":
        train, test = pooled_split(df)
        results, _ = run_experiment(
            "C_accuracy_tuned", train, test, os.path.join(outdir, "C_accuracy_tuned"),
            "sensitivity: same pooled household-grouped split as A, unweighted classes, tuned on accuracy",
            cv=cv, quick=quick, verbose=verbose, interpret=False, scoring="accuracy", class_weighting=False)
        summary["C_accuracy_tuned"] = results

    if verbose:
        print("\n===== Summary: held-out accuracy / macro F1 / ROC AUC =====")
        for exp, results in summary.items():
            for r in results:
                print(f"  {exp:<11} {r['Model']:<20} {r['Accuracy']:.3f} / {r['Macro F1']:.3f} / {r['ROC AUC']:.3f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the digital exclusion experiments.")
    parser.add_argument("--experiment", choices=["both", "A", "B", "C"], default="both",
                        help="both = protocol experiments B then A; C = accuracy-tuned sensitivity run")
    parser.add_argument("--cv", type=int, default=5, help="k for the training-only cross-validation")
    parser.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.experiment, args.cv, args.quick, args.outdir)
