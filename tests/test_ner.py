"""Tests for Entity Extraction (Standard spaCy + Domain Regex)."""

from src.ner.extractor import EntityExtractor


def test_domain_entity_order_id():
    extractor = EntityExtractor()
    text = "My order #ORD45821 is missing and ORD99120 is delayed."
    result = extractor.extract_domain_entities(text)

    order_labels = [e["label"] for e in result]
    assert "ORDER_ID" in order_labels
    assert any("ORD45821" in e["text"] for e in result)


def test_domain_entity_amount_and_email():
    extractor = EntityExtractor()
    text = "I was debited ₹2,499. Contact me at customer.care@example.com or +91-9876543210."
    result = extractor.extract_domain_entities(text)

    labels = {e["label"]: e["text"] for e in result}
    assert "AMOUNT" in labels
    assert "₹2,499" in labels["AMOUNT"]
    assert "EMAIL" in labels
    assert "customer.care@example.com" in labels["EMAIL"]


def test_extract_all_entities():
    extractor = EntityExtractor()
    text = "Order ORD88214 for ₹1,500 was placed on Monday by Rahul."
    res = extractor.extract_all(text)

    assert "all_entities" in res
    assert "domain_entities" in res
    assert "standard_entities" in res
    assert res["total_count"] > 0
