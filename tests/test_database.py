"""Tests for SQLite database operations."""

import tempfile
from pathlib import Path
from src.database.db import DatabaseManager


def test_database_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_resolve.db"
        db = DatabaseManager(db_path)

        # 1. Insert complaint
        cid = db.insert_complaint({
            "complaint_id": "CMP-TEST01",
            "complaint_text": "Refund not credited for order #ORD99120.",
            "category": "Refund Issue",
            "sentiment": "Negative",
            "sentiment_score": -0.65,
            "priority": "High",
            "priority_reasons": ["Unfulfilled refund promise"],
            "summary": "Customer waiting for refund.",
            "entities": [{"label": "ORDER_ID", "text": "ORD99120"}],
            "department": "Finance & Accounts Receivable",
            "recommended_action": "Check bank gateway",
            "suggested_response": "We are reviewing your refund.",
        })

        assert cid == "CMP-TEST01"
        assert db.count_complaints() == 1

        # 2. Fetch by ID
        rec = db.get_complaint_by_id("CMP-TEST01")
        assert rec is not None
        assert rec["category"] == "Refund Issue"
        assert rec["priority"] == "High"
        assert len(rec["entities"]) == 1

        # 3. Query with filter
        filtered = db.get_complaints(category="Refund Issue")
        assert len(filtered) == 1

        # 4. Stats
        stats = db.get_dashboard_stats()
        assert stats["total_complaints"] == 1
        assert stats["high_priority_count"] == 1
        assert stats["negative_count"] == 1
        assert stats["top_category"] == "Refund Issue"

        # 5. Delete
        assert db.delete_complaint("CMP-TEST01") is True
        assert db.count_complaints() == 0
