"""
Streamlit front-end for the credit-risk model bench.

Loads the six classifiers already trained by model/train_models.py, lets a
grader upload a test CSV, pick which classifier to try, and see how it did:
the 6 required metrics, a confusion matrix, and a classification report.

Dataset: UCI Statlog (German Credit Data), numeric version.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STORE_DIR = os.path.join(APP_DIR, "model", "saved_models")
LABEL_COLUMN = "CreditRisk"

MODEL_FILENAME_LOOKUP = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "kNN": "kNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest (Ensemble)": "Random_Forest.pkl",
    "SVM": "SVM.pkl",
}

st.set_page_config(page_title="Credit Risk Classifier", layout="wide")


@st.cache_resource
def load_trained_artifacts():
    """Pull the fitted scaler, the input-column order, and all 6 pickled classifiers off disk."""
    with open(os.path.join(MODEL_STORE_DIR, "scaler.pkl"), "rb") as f:
        feature_scaler = pickle.load(f)
    with open(os.path.join(MODEL_STORE_DIR, "feature_columns.pkl"), "rb") as f:
        input_columns = pickle.load(f)
    trained_models = {}
    for display_name, pkl_filename in MODEL_FILENAME_LOOKUP.items():
        with open(os.path.join(MODEL_STORE_DIR, pkl_filename), "rb") as f:
            trained_models[display_name] = pickle.load(f)
    return feature_scaler, input_columns, trained_models


def score_predictions(true_labels, predicted_labels, positive_class_scores):
    """Bundle the 6 metrics the assignment asks for into one dict."""
    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "AUC": roc_auc_score(true_labels, positive_class_scores),
        "Precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "Recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "F1": f1_score(true_labels, predicted_labels, zero_division=0),
        "MCC": matthews_corrcoef(true_labels, predicted_labels),
    }


def render_app():
    st.title("Credit Risk Classification - Model Demo")
    st.caption(
        "UCI Statlog German Credit Data | 24 numeric features | "
        "Binary classification: Good credit risk (1) vs Bad credit risk (0)"
    )

    feature_scaler, input_columns, trained_models = load_trained_artifacts()

    st.sidebar.header("1. Upload test data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv (must contain the 24 Attribute_* columns "
        "and the 'CreditRisk' label column)",
        type=["csv"],
    )

    st.sidebar.header("2. Choose a model")
    selected_model_name = st.sidebar.selectbox("Model", list(trained_models.keys()))

    compare_all_models = st.sidebar.checkbox(
        "Also compare all 6 models on this data", value=False
    )

    if uploaded_file is None:
        st.info("Upload the test_data.csv file from the sidebar to run predictions.")
        st.markdown(
            "You can use the `test_data.csv` included in this repository "
            "(a 200-row held-out split of the German Credit dataset)."
        )
        return

    uploaded_df = pd.read_csv(uploaded_file)

    missing_columns = [c for c in input_columns if c not in uploaded_df.columns]
    if missing_columns:
        st.error(f"Uploaded CSV is missing required columns: {missing_columns}")
        return
    if LABEL_COLUMN not in uploaded_df.columns:
        st.error(f"Uploaded CSV must contain the '{LABEL_COLUMN}' label column.")
        return

    input_rows = uploaded_df[input_columns]
    true_labels = uploaded_df[LABEL_COLUMN]
    scaled_inputs = feature_scaler.transform(input_rows)

    st.subheader(f"Predictions using: {selected_model_name}")
    chosen_model = trained_models[selected_model_name]
    predicted_labels = chosen_model.predict(scaled_inputs)
    positive_class_scores = (
        chosen_model.predict_proba(scaled_inputs)[:, 1]
        if hasattr(chosen_model, "predict_proba")
        else chosen_model.decision_function(scaled_inputs)
    )

    metric_values = score_predictions(true_labels, predicted_labels, positive_class_scores)

    metric_columns = st.columns(6)
    for column, (metric_name, metric_value) in zip(metric_columns, metric_values.items()):
        column.metric(metric_name, f"{metric_value:.3f}")

    plot_col, report_col = st.columns(2)
    with plot_col:
        st.markdown("**Confusion Matrix**")
        confusion_grid = confusion_matrix(true_labels, predicted_labels)
        fig, axis = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            confusion_grid, annot=True, fmt="d", cmap="Blues", ax=axis,
            xticklabels=["Bad (0)", "Good (1)"],
            yticklabels=["Bad (0)", "Good (1)"],
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        st.pyplot(fig)

    with report_col:
        st.markdown("**Classification Report**")
        report_dict = classification_report(
            true_labels, predicted_labels, target_names=["Bad (0)", "Good (1)"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report_dict).T.round(3))

    with st.expander("View uploaded data"):
        st.dataframe(uploaded_df)

    if compare_all_models:
        st.subheader("Comparison across all 6 models on this uploaded data")
        all_model_scores = {}
        for model_name, model_obj in trained_models.items():
            preds = model_obj.predict(scaled_inputs)
            scores = (
                model_obj.predict_proba(scaled_inputs)[:, 1]
                if hasattr(model_obj, "predict_proba")
                else model_obj.decision_function(scaled_inputs)
            )
            all_model_scores[model_name] = score_predictions(true_labels, preds, scores)
        comparison_table = pd.DataFrame(all_model_scores).T.round(3)
        st.dataframe(comparison_table)
        st.bar_chart(comparison_table[["Accuracy", "AUC", "F1", "MCC"]])


if __name__ == "__main__":
    render_app()
