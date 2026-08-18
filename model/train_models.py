"""
Credit-risk model bench: trains six classifiers on the UCI German Credit
(numeric) data, scores each one on a held-out split, and persists the fitted
classifiers + scaler so the Streamlit demo can reuse them without retraining.

Dataset : UCI Statlog (German Credit Data) - numeric version
Source  : https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Rows    : 1000
Columns : 24 numeric attributes (the original 20 attributes - 7 numeric,
          13 categorical - indicator-coded into 24 numeric columns) plus
          1 label column.
Label   : 1 = Good credit risk, 2 = Bad credit risk in the raw file; we flip
          this to 1/0 below so "Good" is the positive class.

Run from the project root:
    python model/train_models.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "german.data-numeric")
MODEL_STORE_DIR = os.path.join(PROJECT_ROOT, "model", "saved_models")
HOLDOUT_CSV_PATH = os.path.join(PROJECT_ROOT, "test_data.csv")
SCOREBOARD_CSV_PATH = os.path.join(PROJECT_ROOT, "model", "metrics_comparison.csv")

INPUT_COLUMNS = [f"Attribute_{i}" for i in range(1, 25)]
LABEL_COLUMN = "CreditRisk"  # 1 = Good, 0 = Bad after recoding


def read_credit_records() -> pd.DataFrame:
    """Parse the whitespace-separated UCI file and flip the label to 1=Good/0=Bad."""
    records = pd.read_csv(RAW_DATA_FILE, sep=r"\s+", header=None, engine="python")
    records.columns = INPUT_COLUMNS + ["OriginalLabel"]
    # UCI ships 1=Good, 2=Bad; we want the "desirable" outcome (Good) to be the
    # positive class (1) so precision/recall/AUC read intuitively.
    records[LABEL_COLUMN] = (records["OriginalLabel"] == 1).astype(int)
    records = records.drop(columns=["OriginalLabel"])
    return records


def build_classifier_bank() -> dict:
    """One instance per algorithm family the assignment asks for, same hyper-params for everyone."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=SEED
        ),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
    }


def score_classifier(clf, holdout_inputs, holdout_labels) -> dict:
    """Run one trained classifier against the held-out split and collect the 6 required metrics."""
    predicted_labels = clf.predict(holdout_inputs)
    if hasattr(clf, "predict_proba"):
        positive_class_scores = clf.predict_proba(holdout_inputs)[:, 1]
    else:
        positive_class_scores = clf.decision_function(holdout_inputs)
    return {
        "Accuracy": accuracy_score(holdout_labels, predicted_labels),
        "AUC": roc_auc_score(holdout_labels, positive_class_scores),
        "Precision": precision_score(holdout_labels, predicted_labels, zero_division=0),
        "Recall": recall_score(holdout_labels, predicted_labels, zero_division=0),
        "F1": f1_score(holdout_labels, predicted_labels, zero_division=0),
        "MCC": matthews_corrcoef(holdout_labels, predicted_labels),
    }


def run_training_pipeline():
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)

    records = read_credit_records()
    inputs = records[INPUT_COLUMNS]
    labels = records[LABEL_COLUMN]

    train_inputs, holdout_inputs, train_labels, holdout_labels = train_test_split(
        inputs, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    feature_scaler = StandardScaler()
    train_inputs_scaled = feature_scaler.fit_transform(train_inputs)
    holdout_inputs_scaled = feature_scaler.transform(holdout_inputs)

    with open(os.path.join(MODEL_STORE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(MODEL_STORE_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(INPUT_COLUMNS, f)

    scoreboard = {}
    for clf_name, clf in build_classifier_bank().items():
        clf.fit(train_inputs_scaled, train_labels)
        scoreboard[clf_name] = score_classifier(clf, holdout_inputs_scaled, holdout_labels)

        # Strip the "(Ensemble)" suffix and spaces so the pickle filename stays filesystem-friendly.
        file_stub = clf_name.split(" (")[0].replace(" ", "_")
        with open(os.path.join(MODEL_STORE_DIR, f"{file_stub}.pkl"), "wb") as f:
            pickle.dump(clf, f)

    scoreboard_table = pd.DataFrame(scoreboard).T
    scoreboard_table.index.name = "ML Model Name"
    scoreboard_table = scoreboard_table.round(4)
    scoreboard_table.to_csv(SCOREBOARD_CSV_PATH)

    print("\n=== Evaluation metrics on the held-out test split (20% of data) ===\n")
    print(scoreboard_table.to_string())

    # Ship the held-out rows (original feature values + true label, unscaled)
    # as the CSV a grader can upload straight into the Streamlit app.
    holdout_export = holdout_inputs.copy()
    holdout_export[LABEL_COLUMN] = holdout_labels.values
    holdout_export.to_csv(HOLDOUT_CSV_PATH, index=False)
    print(f"\nSaved test_data.csv with {len(holdout_export)} rows -> {HOLDOUT_CSV_PATH}")
    print(f"Saved metrics table -> {SCOREBOARD_CSV_PATH}")
    print(f"Saved models + scaler -> {MODEL_STORE_DIR}")


if __name__ == "__main__":
    run_training_pipeline()
