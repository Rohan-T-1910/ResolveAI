"""Complaint History View for ResolveAI.
Allows support agents to search, filter, inspect, and manage historical complaint records.
"""

import streamlit as st
import pandas as pd
from src.database.db import DatabaseManager
from src.config import DB_PATH, CATEGORIES, PRIORITY_LEVELS, SENTIMENT_LABELS
from src.styles import get_priority_badge, get_sentiment_badge


def render_history():
    """Renders the Complaint History and Log Explorer page."""
    st.title("📁 Complaint History")
    st.markdown("Search, filter, and inspect past customer complaint analyses stored in the SQLite database.")

    db = DatabaseManager(DB_PATH)

    # Filter Bar
    f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1])
    with f1:
        search_query = st.text_input("🔍 Search Text or ID:", placeholder="e.g. ORD45821, refund, delayed...")
    with f2:
        cat_filter = st.selectbox("Category:", ["All"] + CATEGORIES)
    with f3:
        sent_filter = st.selectbox("Sentiment:", ["All"] + SENTIMENT_LABELS)
    with f4:
        prio_filter = st.selectbox("Priority:", ["All"] + PRIORITY_LEVELS)

    # Query database
    complaints = db.get_complaints(
        search=search_query,
        category=cat_filter,
        sentiment=sent_filter,
        priority=prio_filter,
        limit=100,
    )

    st.markdown(f"**Found {len(complaints)} complaint(s)**")

    if not complaints:
        st.info("No complaints matching the current search criteria.")
        return

    # Display as interactive list
    for item in complaints:
        with st.expander(f"{item['complaint_id']} — {item['category']} ({item['priority']} Priority) — {item['created_at']}"):
            st.markdown(f"**Original Complaint:**\n> {item['complaint_text']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Category:** {item['category']}")
            c2.markdown(f"**Priority:** {get_priority_badge(item['priority'])}", unsafe_allow_html=True)
            c3.markdown(f"**Sentiment:** {get_sentiment_badge(item['sentiment'], item['sentiment_score'])}", unsafe_allow_html=True)
            c4.markdown(f"**Department:** {item['department']}")

            st.markdown(f"**Extractive Summary:** {item['summary']}")

            if item.get("priority_reasons"):
                st.markdown("**Priority Signals:**")
                for r in item["priority_reasons"]:
                    st.markdown(f"- {r}")

            if item.get("entities"):
                st.markdown("**Entities:**")
                ent_str = " ".join(f"`{e['label']}:{e['text']}`" for e in item["entities"])
                st.markdown(ent_str)

            st.markdown(f"**Recommended Action:**\n{item['recommended_action']}")
            st.markdown(f"**Suggested Response:**\n*{item['suggested_response']}*")

            b_col1, b_col2 = st.columns([1.5, 4])
            with b_col1:
                if st.button("Load in Analyzer", key=f"load_{item['complaint_id']}"):
                    st.session_state["complaint_input"] = item["complaint_text"]
                    st.session_state["auto_analyze"] = True
                    st.session_state["nav_selection"] = "Complaint Analyzer"
                    st.rerun()
            with b_col2:
                if st.button("🗑️ Delete Record", key=f"del_{item['complaint_id']}"):
                    db.delete_complaint(item["complaint_id"])
                    st.success(f"Deleted {item['complaint_id']}")
                    st.rerun()
