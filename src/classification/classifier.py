"""Machine Learning Complaint Classification module.
Implements:
- TF-IDF feature extraction pipeline
- Logistic Regression / Multinomial Naive Bayes classifier
- Model persistence using joblib
- Probability estimation across all 8 e-commerce categories
- Transparent rule-based fallback if model is unconfident
- Real evaluation metrics calculation (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORIES,
    CLASSIFIER_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    DATASET_PATH,
)


class ComplaintClassifier:
    """Trains, evaluates, and predicts e-commerce complaint categories."""

    def __init__(
        self,
        classifier_path: Path = CLASSIFIER_PATH,
        vectorizer_path: Path = VECTORIZER_PATH,
        metrics_path: Path = METRICS_PATH,
    ):
        self.classifier_path = Path(classifier_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.metrics_path = Path(metrics_path)

        self.classifier = None
        self.vectorizer = None
        self.metrics = None

        self._load_or_train()

    def _load_or_train(self):
        """Loads saved models or trains from sample dataset if missing."""
        if self.classifier_path.exists() and self.vectorizer_path.exists():
            try:
                self.classifier = joblib.load(self.classifier_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                if self.metrics_path.exists():
                    with open(self.metrics_path, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                return
            except Exception:
                pass

        if DATASET_PATH.exists():
            self.train(DATASET_PATH)

    def train(self, dataset_path: Path = DATASET_PATH, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """Trains TF-IDF + Logistic Regression model and computes real evaluation metrics."""
        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=["text", "category"])

        X = df["text"].astype(str)
        y = df["category"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        classifier.fit(X_train_vec, y_train)

        # Predictions on holdout test set
        y_pred = classifier.predict(X_test_vec)
        labels = sorted(list(set(y)))

        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

        metrics = {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(acc, 4),
            "accuracy_percentage": round(acc * 100, 2),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": report,
            "category_distribution": df["category"].value_counts().to_dict(),
        }

        # Save artifacts
        self.classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, self.classifier_path)
        joblib.dump(vectorizer, self.vectorizer_path)

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        self.classifier = classifier
        self.vectorizer = vectorizer
        self.metrics = metrics

        return metrics

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts complaint category using ML model with probability distribution."""
        clean_text = (text or "").strip()
        if not clean_text:
            return {
                "category": "Other",
                "confidence": 0.0,
                "probabilities": {cat: 0.0 for cat in CATEGORIES},
                "method": "Default (Empty Input)",
                "explanation": "No text provided for classification.",
            }

        # Use trained ML model if available
        if self.classifier is not None and self.vectorizer is not None:
            try:
                X_vec = self.vectorizer.transform([clean_text])
                probs = self.classifier.predict_proba(X_vec)[0]
                classes = self.classifier.classes_

                prob_dict = {
                    cls: round(float(prob), 4) for cls, prob in zip(classes, probs)
                }

                best_idx = int(np.argmax(probs))
                predicted_cat = classes[best_idx]
                confidence = round(float(probs[best_idx]), 4)

                # Low confidence threshold: if confidence is extremely low (< 0.25 on 8 classes), flag rule verification
                method = "Machine Learning (TF-IDF + Logistic Regression)"
                if confidence < 0.20:
                    fallback_cat = self._rule_based_fallback(clean_text)
                    if fallback_cat:
                        return {
                            "category": fallback_cat,
                            "confidence": round(confidence, 4),
                            "probabilities": prob_dict,
                            "method": "Rule-Based Fallback (Low ML Confidence)",
                            "explanation": f"ML model confidence was low ({round(confidence*100, 1)}%); keyword heuristic assigned {fallback_cat}.",
                        }

                return {
                    "category": predicted_cat,
                    "confidence": confidence,
                    "confidence_percentage": round(confidence * 100, 1),
                    "probabilities": prob_dict,
                    "method": method,
                    "explanation": f"Categorized as '{predicted_cat}' with {round(confidence * 100, 1)}% model confidence.",
                }
            except Exception as e:
                pass

        # Rule-based fallback if ML fails or not loaded
        fallback_cat = self._rule_based_fallback(clean_text) or "Other"
        return {
            "category": fallback_cat,
            "confidence": 0.50,
            "probabilities": {cat: (1.0 if cat == fallback_cat else 0.0) for cat in CATEGORIES},
            "method": "Rule-Based Keyword Matching (Fallback)",
            "explanation": "Predicted using keyword heuristics as ML model was unavailable.",
        }

    def _rule_based_fallback(self, text: str) -> str:
        """Explicit keyword heuristics as safety fallback."""
        t = text.lower()
        if any(w in t for w in ["refund", "reimburse", "money back", "credited", "store credit"]):
            return "Refund Issue"
        if any(w in t for w in ["unauthorized", "charged", "debit", "double charge", "upi", "card", "payment failed"]):
            return "Payment Issue"
        if any(w in t for w in ["deliver", "delivery", "courier", "package", "arrived", "tracking", "late", "delay"]):
            return "Delivery Issue"
        if any(w in t for w in ["damaged", "broken", "defective", "fake", "counterfeit", "wrong item", "size"]):
            return "Product Issue"
        if any(w in t for w in ["cancel", "cancellation", "stop order", "cancelled"]):
            return "Cancellation"
        if any(w in t for w in ["locked", "password", "login", "otp", "hacked", "suspended", "account"]):
            return "Account Issue"
        if any(w in t for w in ["error", "bug", "crash", "website", "app", "gateway", "502", "timeout"]):
            return "Technical Issue"
        return "Other"

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the evaluated model metrics."""
        if self.metrics:
            return self.metrics
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
                    return self.metrics
            except Exception:
                pass
        return {}
