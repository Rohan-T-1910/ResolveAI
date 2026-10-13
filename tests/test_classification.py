"""Tests for complaint classification."""

from src.classification.classifier import ComplaintClassifier
from src.config import CATEGORIES


def test_classifier_prediction():
    clf = ComplaintClassifier()
    text = "My order #ORD45821 was supposed to arrive yesterday but it still hasn't arrived. The courier is late."
    result = clf.predict(text)

    assert result["category"] in CATEGORIES
    assert result["confidence"] > 0.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == len(CATEGORIES)


def test_classifier_refund_prediction():
    clf = ComplaintClassifier()
    text = "I cancelled my order and customer care promised a refund of ₹2500 but money is not credited."
    result = clf.predict(text)

    assert result["category"] == "Refund Issue"


def test_classifier_empty():
    clf = ComplaintClassifier()
    result = clf.predict("")
    assert result["category"] == "Other"
