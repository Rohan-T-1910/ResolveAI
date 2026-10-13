"""Tests for Priority Engine."""

from src.priority.engine import PriorityEngine


def test_critical_fraud_priority():
    engine = PriorityEngine()
    text = "Someone made an unauthorized transaction of ₹5000 on my card. This is fraud! Block account."
    result = engine.determine_priority(text, category="Payment Issue", sentiment="Negative", compound_score=-0.7)

    assert result["priority"] == "Critical"
    assert any("fraud" in r.lower() or "security" in r.lower() for r in result["reasons"])


def test_high_priority_delivery():
    engine = PriorityEngine()
    text = "My order #ORD45821 was supposed to arrive yesterday. No response from courier. Extremely frustrating."
    result = engine.determine_priority(text, category="Delivery Issue", sentiment="Negative", compound_score=-0.6)

    assert result["priority"] in ["High", "Critical"]
    assert len(result["reasons"]) >= 2


def test_low_priority_inquiry():
    engine = PriorityEngine()
    text = "Could you please clarify your return policy for electronics?"
    result = engine.determine_priority(text, category="Other", sentiment="Neutral", compound_score=0.1)

    assert result["priority"] == "Low"
