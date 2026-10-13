"""Tests for Resolution and Response Engine."""

from src.resolution.engine import ResolutionEngine


def test_resolution_recommendation():
    engine = ResolutionEngine()
    entities = [{"label": "ORDER_ID", "text": "ORD45821"}, {"label": "AMOUNT", "text": "₹2,499"}]
    res = engine.generate_recommendation(
        category="Delivery Issue",
        priority="High",
        entities=entities,
        sentiment="Negative",
    )

    assert "Logistics" in res["department"]
    assert "tracking" in res["recommended_action"].lower()
    assert "ORD45821" in res["suggested_response"]
    # Check strict guardrail: never claims completed action
    assert "has been processed" not in res["suggested_response"]
    assert "has already arrived" not in res["suggested_response"]


def test_positive_feedback_recommendation():
    engine = ResolutionEngine()
    entities = [{"label": "ORDER_ID", "text": "ORD99881"}]
    res = engine.generate_recommendation(
        category="Delivery Issue",
        priority="Low",
        entities=entities,
        sentiment="Positive",
    )

    assert "No action required" in res["recommended_action"]
    assert "ORD99881" in res["suggested_response"]
    assert "thank you" in res["suggested_response"].lower()
    assert any("positive feedback" in note.lower() for note in res["agent_notes"])
