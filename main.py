# Main Python File: main.py
# The purpose of this file is to run all phases of the experiment
# including data preparation, model training, and model testing. It also compares the performance of
# different models for each disability type.

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

from xgboost import XGBClassifier

import dataPrep

PREDICTOR_FILE = "data/predictor_data.csv"

# nominal census codes (the number itself has no order, e.g. state FIPS) -> one-hot encoded
CATEGORICAL_FEATURES = ["PESEX", "PTDTRACE", "PEMARITL", "HETENURE", "GESTFIPS", "PEMLR"]

# age and ordered codes (income bracket, education level) -> scaled
NUMERIC_FEATURES = ["PRTAGE", "HEFAMINC", "PEEDUCA"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# one binary classification problem per disability type
TARGETS = ["DIS_HEARING", "DIS_VISION", "DIS_COGNITIVE", "DIS_PHYSICAL", "DIS_SELFCARE", "DIS_INDLIVING"]

# metric used both to tune hyperparameters and to pick the best model
# (macro F1 weighs both classes equally, so predicting only the majority class scores poorly)
SCORING = "f1_macro"


# Preprocessing: fit on training folds only because it lives inside each model's Pipeline
def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


# Each model with class-imbalance handling and a small hyperparameter grid
def build_models(y_train):
    # XGBoost has no class_weight option; weight positives by the negative/positive ratio instead
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [1, 20, 50]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 20]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, n_jobs=1, scale_pos_weight=pos_weight),
            {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1], "model__n_estimators": [100, 300]},
        ),
    }


def train_models(df=None, target="DIS_PHYSICAL", cv=5):

    # read in data
    if df is None:
        df = pd.read_csv(PREDICTOR_FILE)

    x = df[FEATURES]
    y = df[target]  # Target variable

    # stratify so train and test keep the same class ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Tune each model with cross-validation on the training data only
    models = {}
    for name, (estimator, param_grid) in build_models(y_train).items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(pipeline, param_grid, scoring=SCORING, cv=folds, n_jobs=-1)
        search.fit(x_train, y_train)

        models[name] = search.best_estimator_
        train_acc = accuracy_score(y_train, search.best_estimator_.predict(x_train))
        print(f"{name}: best params {search.best_params_}, CV {SCORING}={search.best_score_:.3f}, "
              f"train accuracy={train_acc:.3f}")

    print(" Model Training Complete")
    return models, x_train, x_test, y_train, y_test


def test_models(models, x_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]  # Probability of the positive class

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Macro F1": f1_score(y_test, y_pred, average="macro"),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, probabilities),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, zero_division=0),
        }

        results.append(metrics)

    print(" Model Testing Complete. Initiating Model Comparison...\n\n")

    # Majority-class baseline: a useful model has to beat this
    majority = y_test.mode()[0]
    baseline_pred = pd.Series(majority, index=y_test.index)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_macro_f1 = f1_score(y_test, baseline_pred, average="macro")

    # Compare models by macro F1
    best_model = max(results, key=lambda r: r["Macro F1"])

    print("Model Comparison:")
    print(f"Baseline (always {majority}): Accuracy={baseline_acc:.3f}, Balanced Acc=0.500, "
          f"Macro F1={baseline_macro_f1:.3f}, AUC=0.500")
    for r in results:
        print(f"{r['Model']}: Accuracy={r['Accuracy']:.3f}, Balanced Acc={r['Balanced Accuracy']:.3f}, "
              f"Macro F1={r['Macro F1']:.3f}, AUC={r['ROC AUC']:.3f}")

    print(f"\nBest model: {best_model['Model']} with Macro F1={best_model['Macro F1']:.3f}")
    print(best_model["Classification Report"])

    return results, best_model


def main():

    # Step 1: Data Preparation (skipped when the predictor file already exists; delete it to rebuild)
    if not Path(PREDICTOR_FILE).exists():
        if dataPrep.main() is None:
            return None

    summary = []
    for target in TARGETS:
        print(f"\n===== Target: {target} =====")

        # Step 2: Model Training
        models, x_train, x_test, y_train, y_test = train_models(target=target)

        # Step 3: Model Testing
        results, best_model = test_models(models, x_test, y_test)
        summary.append({
            "Target": target,
            "Best Model": best_model["Model"],
            "Macro F1": round(best_model["Macro F1"], 3),
            "Balanced Accuracy": round(best_model["Balanced Accuracy"], 3),
            "ROC AUC": round(best_model["ROC AUC"], 3),
        })

    summary = pd.DataFrame(summary)
    print("\n===== Summary: best model per disability type =====")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
