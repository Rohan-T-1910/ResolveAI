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
