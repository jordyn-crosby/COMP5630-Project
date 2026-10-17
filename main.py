# Main Python File: main.py
# The purpose of this file is to run all of the stages of this experiment

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, \
    precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

import dataPrep

# Supervised Classifier Function
def supervised_classifier(model, x_train, y_train, x_test, y_test):

    # 1. Fit model on the training data
    model.fit(x_train, y_train)

    # 2. Predict on unseen test data
    y_pred = model.predict(x_test)

    # 3. Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 4. Probability Output
    probabilities = model.predict_proba(x_test)[:, 1]  # Probability of the positive class

    return{
        "Model": model,
        "Accuracy": accuracy,
        "Predictions": y_pred,
        "Classification Report": report,
        "Probabilities": probabilities
    }

def train_models():

    # read in data
    df = pd.read_csv("data/predictor_data.csv")

    # Logistic Regression
    x = df[["PRTAGE", "PESEX", "PTDTRACE", "PEMARITL", "HEFAMINC",
            "PEEDUCA", "HETENURE", "GESTFIPS", "PEMLR"]]

    y = df["DIS_PHYSICAL"]  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model_lr = LogisticRegression(max_iter=1000, solver="liblinear")
    results_lr = supervised_classifier(model_lr, x_train, y_train, x_test, y_test)
    print("Logistic Regression Results:")
    print(results_lr["Accuracy"])
    print(results_lr["Classification Report"])

    # Decision Tree
    model_tree = DecisionTreeClassifier(random_state=42)
    results_tree = supervised_classifier(model_tree, x_train, y_train, x_test, y_test)
    print("Decision Tree Results:")
    print(results_tree["Accuracy"])
    print(results_tree["Classification Report"])

    # Random Forest
    model_rf = RandomForestClassifier(random_state=42)
    results_rf = supervised_classifier(model_rf, x_train, y_train, x_test, y_test)
    print("Random Forest Results:")
    print(results_rf["Accuracy"])
    print(results_rf["Classification Report"])

    # XGBoost
    model_xgb = XGBClassifier(random_state=42)
    results_xgb = supervised_classifier(model_xgb, x_train, y_train, x_test, y_test)
    print("XGBoost Results:")
    print(results_xgb["Accuracy"])
    print(results_xgb["Classification Report"])

    return 0

def main():
    # Step 1: Data Preparation
    dataPrep.main()

    # Step 2: Model Training and Evaluation
    train_models()


if __name__ == "__main__":
    main()