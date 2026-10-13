"""Model Evaluation View for ResolveAI.
Visualizes real statistical metrics, classification report, and confusion matrix
evaluated on the holdout test set from the trained model.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.classification.classifier import ComplaintClassifier
from src.config import DATASET_PATH


def render_evaluation(classifier: ComplaintClassifier):
    """Renders the Model Evaluation page."""
    st.title("📈 Model Evaluation & Performance")
    st.markdown(
        "Empirical performance evaluation of the TF-IDF + Logistic Regression classifier on holdout test data."
    )

    metrics = classifier.get_metrics()
    if not metrics:
        st.warning("No evaluation metrics found. Please train the model first.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training classification model..."):
                metrics = classifier.train(DATASET_PATH)
            st.success("Model trained successfully!")
            st.rerun()
        return

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{metrics.get('accuracy_percentage', 0)}%")
    m2.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
    m3.metric("Weighted F1-Score", f"{metrics.get('weighted_f1', 0):.4f}")
    m4.metric("Test Samples", f"{metrics.get('test_samples', 0)} / {metrics.get('total_samples', 0)}")

    st.markdown("---")

    # Retrain button
    with st.expander("⚙️ Retrain Classifier Model"):
        st.markdown(
            "Retrain the TF-IDF feature extractor and Logistic Regression model on the current dataset."
        )
        if st.button("🔄 Retrain and Recompute Metrics", type="primary"):
            with st.spinner("Retraining model and evaluating test split..."):
                new_metrics = classifier.train(DATASET_PATH)
            st.success("Retrained successfully!")
            st.rerun()

    # Classification Report Table
    st.subheader("Detailed Category Performance (Holdout Test Set)")
    report = metrics.get("classification_report", {})
    labels = metrics.get("labels", [])

    rows = []
    for lbl in labels:
        if lbl in report:
            rows.append({
                "Category": lbl,
                "Precision": round(float(report[lbl]["precision"]), 4),
                "Recall": round(float(report[lbl]["recall"]), 4),
                "F1-Score": round(float(report[lbl]["f1-score"]), 4),
                "Support (Test Count)": int(report[lbl]["support"]),
            })

    if rows:
        df_rep = pd.DataFrame(rows)
        st.dataframe(df_rep, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    cm = metrics.get("confusion_matrix", [])
    if cm and labels:
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Dataset Distribution Note
    st.info(
        "Note on Evaluation: The dataset consists of 320 labeled e-commerce complaints balanced across "
        "8 categories (40 samples per category) with an 80/20 train/test split. Stratified sampling is used "
        "to ensure fair class representation."
    )
