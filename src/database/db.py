"""SQLite Database Layer for ResolveAI.
Manages persistent storage, historical retrieval, filtering, and real-time dashboard analytics.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import DB_PATH


class DatabaseManager:
    """Handles all SQLite database interactions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates and returns a connection to SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema if tables do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS complaints (
                    complaint_id TEXT PRIMARY KEY,
                    complaint_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    priority TEXT NOT NULL,
                    priority_reasons TEXT,
                    summary TEXT,
                    entities TEXT,
                    department TEXT,
                    recommended_action TEXT,
                    suggested_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cat ON complaints(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent ON complaints(sentiment)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prio ON complaints(priority)")
            conn.commit()

    def insert_complaint(self, data: Dict[str, Any]) -> str:
        """Inserts a processed complaint analysis record into the database."""
        complaint_id = data.get("complaint_id") or f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

        reasons_json = json.dumps(data.get("priority_reasons", []))
        entities_json = json.dumps(data.get("entities", []))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO complaints (
                    complaint_id, complaint_text, category, sentiment, sentiment_score,
                    priority, priority_reasons, summary, entities, department,
                    recommended_action, suggested_response, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                complaint_id,
                data.get("complaint_text", ""),
                data.get("category", "Other"),
                data.get("sentiment", "Neutral"),
                float(data.get("sentiment_score", 0.0)),
                data.get("priority", "Low"),
                reasons_json,
                data.get("summary", ""),
                entities_json,
                data.get("department", "Customer Support"),
                data.get("recommended_action", ""),
                data.get("suggested_response", ""),
                data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ))
            conn.commit()

        return complaint_id

    def get_complaints(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Queries complaints with optional keyword search and category/sentiment/priority filters."""
        query = "SELECT * FROM complaints WHERE 1=1"
        params = []

        if search and search.strip():
            query += " AND (complaint_text LIKE ? OR complaint_id LIKE ? OR summary LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        if category and category != "All":
            query += " AND category = ?"
            params.append(category)

        if sentiment and sentiment != "All":
            query += " AND sentiment = ?"
            params.append(sentiment)

        if priority and priority != "All":
            query += " AND priority = ?"
            params.append(priority)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            d = dict(row)
            try:
                d["priority_reasons"] = json.loads(d.get("priority_reasons") or "[]")
            except Exception:
                d["priority_reasons"] = []
            try:
                d["entities"] = json.loads(d.get("entities") or "[]")
            except Exception:
                d["entities"] = []
            results.append(d)

        return results

    def get_complaint_by_id(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single complaint by its unique identifier."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
            row = cursor.fetchone()

        if not row:
            return None

        d = dict(row)
        try:
            d["priority_reasons"] = json.loads(d.get("priority_reasons") or "[]")
        except Exception:
            d["priority_reasons"] = []
        try:
            d["entities"] = json.loads(d.get("entities") or "[]")
        except Exception:
            d["entities"] = []

        return d

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Calculates real aggregate metrics and distributions for the dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total complaints
            cursor.execute("SELECT COUNT(*) FROM complaints")
            total = cursor.fetchone()[0]

            if total == 0:
                return {
                    "total_complaints": 0,
                    "high_priority_count": 0,
                    "negative_count": 0,
                    "top_category": "N/A",
                    "avg_sentiment_score": 0.0,
                    "category_distribution": {},
                    "sentiment_distribution": {},
                    "priority_distribution": {},
                    "recent_complaints": [],
                }

            # High or Critical priority
            cursor.execute("SELECT COUNT(*) FROM complaints WHERE priority IN ('High', 'Critical')")
            high_priority_count = cursor.fetchone()[0]

            # Negative sentiment
            cursor.execute("SELECT COUNT(*) FROM complaints WHERE sentiment = 'Negative'")
            negative_count = cursor.fetchone()[0]

            # Average sentiment score
            cursor.execute("SELECT AVG(sentiment_score) FROM complaints")
            avg_sentiment = cursor.fetchone()[0] or 0.0

            # Category distribution
            cursor.execute("SELECT category, COUNT(*) as cnt FROM complaints GROUP BY category ORDER BY cnt DESC")
            cat_rows = cursor.fetchall()
            cat_dist = {r["category"]: r["cnt"] for r in cat_rows}
            top_category = cat_rows[0]["category"] if cat_rows else "N/A"

            # Sentiment distribution
            cursor.execute("SELECT sentiment, COUNT(*) as cnt FROM complaints GROUP BY sentiment")
            sent_rows = cursor.fetchall()
            sent_dist = {r["sentiment"]: r["cnt"] for r in sent_rows}

            # Priority distribution
            cursor.execute("SELECT priority, COUNT(*) as cnt FROM complaints GROUP BY priority")
            prio_rows = cursor.fetchall()
            prio_dist = {r["priority"]: r["cnt"] for r in prio_rows}

            # Recent 10 complaints
            cursor.execute("""
                SELECT complaint_id, category, sentiment, priority, created_at, SUBSTR(complaint_text, 1, 90) as preview
                FROM complaints
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent = [dict(r) for r in cursor.fetchall()]

        return {
            "total_complaints": total,
            "high_priority_count": high_priority_count,
            "negative_count": negative_count,
            "top_category": top_category,
            "avg_sentiment_score": round(float(avg_sentiment), 3),
            "category_distribution": cat_dist,
            "sentiment_distribution": sent_dist,
            "priority_distribution": prio_dist,
            "recent_complaints": recent,
        }

    def count_complaints(self) -> int:
        """Returns total number of complaints currently stored."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM complaints")
            return cursor.fetchone()[0]

    def delete_complaint(self, complaint_id: str) -> bool:
        """Deletes a complaint record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM complaints WHERE complaint_id = ?", (complaint_id,))
            conn.commit()
            return cursor.rowcount > 0
