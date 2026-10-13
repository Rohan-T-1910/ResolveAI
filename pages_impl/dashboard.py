"""Dashboard View for ResolveAI.
Visualizes real-time metrics, distributions, and recent complaint logs from SQLite database.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.database.db import DatabaseManager
from src.config import DB_PATH
from src.styles import get_priority_badge, get_sentiment_badge


def render_dashboard():
    """Renders the executive analytics dashboard."""
    st.title("📊 Support Operations Dashboard")
    st.markdown(
        "Real-time NLP analytics and complaint intelligence across customer support channels."
    )

    db = DatabaseManager(DB_PATH)
    stats = db.get_dashboard_stats()

    if stats["total_complaints"] == 0:
        st.info("No complaints found in the database. Analyze your first complaint or run the database seeder.")
        if st.button("🌱 Seed Database with Sample Complaints"):
            from scripts.seed_db import seed_database
            with st.spinner("Seeding database..."):
                seed_database(30)
            st.rerun()
        return

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{stats['total_complaints']}</div>
                <div class="metric-lbl">Total Complaints</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color: #dc2626;">{stats['high_priority_count']}</div>
                <div class="metric-lbl">High / Critical</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color: #e11d48;">{stats['negative_count']}</div>
                <div class="metric-lbl">Negative Sentiment</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="font-size: 1.25rem; line-height: 2.1rem; color: #2563eb;">{stats['top_category']}</div>
                <div class="metric-lbl">Top Category</div>
            </div>
        """, unsafe_allow_html=True)

    with m5:
        score_val = stats['avg_sentiment_score']
        score_color = "#16a34a" if score_val > 0.05 else ("#dc2626" if score_val < -0.05 else "#64748b")
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color: {score_color};">{score_val:+.2f}</div>
                <div class="metric-lbl">Avg Sentiment Score</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Complaint Category Distribution")
        cat_data = stats["category_distribution"]
        if cat_data:
            df_cat = pd.DataFrame(list(cat_data.items()), columns=["Category", "Count"])
            fig_cat = px.bar(
                df_cat,
                x="Count",
                y="Category",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                text="Count",
            )
            fig_cat.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        st.subheader("Sentiment & Priority Split")
        t1, t2 = st.tabs(["Sentiment", "Priority"])

        with t1:
            sent_data = stats["sentiment_distribution"]
            if sent_data:
                color_map = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
                fig_sent = px.pie(
                    names=list(sent_data.keys()),
                    values=list(sent_data.values()),
                    hole=0.45,
                    color=list(sent_data.keys()),
                    color_discrete_map=color_map,
                )
                fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
                st.plotly_chart(fig_sent, use_container_width=True)

        with t2:
            prio_data = stats["priority_distribution"]
            if prio_data:
                prio_order = ["Critical", "High", "Medium", "Low"]
                prio_colors = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#d97706", "Low": "#16a34a"}
                df_prio = pd.DataFrame(
                    [{"Priority": p, "Count": prio_data.get(p, 0)} for p in prio_order if p in prio_data]
                )
                fig_prio = px.bar(
                    df_prio,
                    x="Priority",
                    y="Count",
                    color="Priority",
                    color_discrete_map=prio_colors,
                    text="Count",
                )
                fig_prio.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280, showlegend=False)
                st.plotly_chart(fig_prio, use_container_width=True)

    # Recent Complaints Table
    st.subheader("Recent Complaints")
    recent = stats["recent_complaints"]
    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent.columns = ["ID", "Category", "Sentiment", "Priority", "Timestamp", "Preview"]
        st.dataframe(
            df_recent[["ID", "Category", "Sentiment", "Priority", "Timestamp", "Preview"]],
            use_container_width=True,
            hide_index=True,
        )
