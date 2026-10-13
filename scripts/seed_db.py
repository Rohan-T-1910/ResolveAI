"""Database Seeding Script for ResolveAI.
Populates SQLite database (data/resolve_ai.db) with historical complaint records
processed through the actual NLP pipeline.
Run:
    python scripts/seed_db.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.config import DATASET_PATH, DB_PATH
from src.pipeline import ResolvePipeline
from src.database.db import DatabaseManager


def seed_database(limit: int = 50):
    print("=" * 60)
    print("ResolveAI: Seeding SQLite Database with NLP-Processed History")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}. Please provide a complaint dataset before seeding.")
        return

    df = pd.read_csv(DATASET_PATH)
    pipeline = ResolvePipeline()
    db = DatabaseManager(DB_PATH)

    current_count = db.count_complaints()
    if current_count >= limit:
        print(f"Database already contains {current_count} records. Skipping seed.")
        return

    sample_df = df.head(limit)
    print(f"Processing and storing {len(sample_df)} sample complaints through NLP pipeline...")

    for idx, row in sample_df.iterrows():
        text = str(row["text"])
        cid = str(row["complaint_id"])

        # Run complaint through the actual pipeline
        analysis = pipeline.analyze_complaint(text, complaint_id=cid)

        # Store in database
        db.insert_complaint({
            "complaint_id": cid,
            "complaint_text": text,
            "category": analysis["category"],
            "sentiment": analysis["sentiment"],
            "sentiment_score": analysis["sentiment_score"],
            "priority": analysis["priority"],
            "priority_reasons": analysis["priority_reasons"],
            "summary": analysis["summary"],
            "entities": analysis["entities"],
            "department": analysis["department"],
            "recommended_action": analysis["recommended_action"],
            "suggested_response": analysis["suggested_response"],
        })

    total = db.count_complaints()
    print(f"✓ Successfully seeded database! Current total complaints in DB: {total}")
    print("=" * 60)


if __name__ == "__main__":
    seed_database()
