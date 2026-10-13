"""Complaint Analyzer View for ResolveAI.
Main workplace for support agents to enter complaints, execute NLP analysis,
inspect structured actionable insights, and interact with the complaint assistant.
"""

import streamlit as st
from datetime import datetime
from src.pipeline import ResolvePipeline
from src.chatbot.assistant import ComplaintAssistant
from src.database.db import DatabaseManager
from src.config import DB_PATH
from src.styles import get_priority_badge, get_sentiment_badge


def render_analyzer(pipeline: ResolvePipeline):
    """Renders the Complaint Analyzer page."""
    st.title("🔍 Intelligent Complaint Analyzer")
    st.markdown(
        "Enter a customer complaint to perform multi-stage NLP analysis, priority triaging, and automated resolution drafting."
    )

    db = DatabaseManager(DB_PATH)
    assistant = ComplaintAssistant()

    # Text Input Form
    default_text = st.session_state.get("complaint_input", "")

    with st.form("analyzer_form"):
        complaint_text = st.text_area(
            "Customer Complaint Text:",
            value=default_text,
            height=140,
            placeholder="Paste or type customer email, chat message, or support ticket here...",
        )
        c1, c2, c3 = st.columns([1.5, 1, 4])
        with c1:
            submitted = st.form_submit_button("⚡ Analyze Complaint", use_container_width=True, type="primary")
        with c2:
            cleared = st.form_submit_button("✕ Clear", use_container_width=True)

    if cleared:
        st.session_state["complaint_input"] = ""
        st.session_state["active_analysis"] = None
        st.rerun()

    # Trigger analysis on submit
    if submitted and complaint_text.strip():
        with st.spinner("Executing multi-stage NLP pipeline..."):
            cid = f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            analysis = pipeline.analyze_complaint(complaint_text, complaint_id=cid)

            # Store in database
            db.insert_complaint({
                "complaint_id": cid,
                "complaint_text": complaint_text,
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

            st.session_state["active_analysis"] = analysis
            st.session_state["active_complaint_id"] = cid

    # Display Analysis Results
    analysis = st.session_state.get("active_analysis")
    if analysis:
        st.markdown("---")
        st.subheader("📋 Actionable Support Report")

        # Top Metric Badges Row
        col_cat, col_prio, col_sent, col_dept = st.columns(4)
        with col_cat:
            conf_str = f" ({analysis['classification'].get('confidence_percentage', 0)}%)" if 'confidence_percentage' in analysis['classification'] else ""
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="font-size: 1.25rem; color: #1e3a8a;">{analysis['category']}</div>
                    <div class="metric-lbl">Complaint Category{conf_str}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_prio:
            prio_badge = get_priority_badge(analysis['priority'])
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="font-size: 1.25rem;">{prio_badge}</div>
                    <div class="metric-lbl">Priority Level</div>
                </div>
            """, unsafe_allow_html=True)

        with col_sent:
            sent_badge = get_sentiment_badge(analysis['sentiment'], analysis['sentiment_score'])
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="font-size: 1.25rem;">{sent_badge}</div>
                    <div class="metric-lbl">Sentiment Tone</div>
                </div>
            """, unsafe_allow_html=True)

        with col_dept:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="font-size: 1.1rem; color: #0f766e;">{analysis['department']}</div>
                    <div class="metric-lbl">Assigned Department</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary and Priority Reasons
        r1, r2 = st.columns([1.2, 1])
        with r1:
            st.markdown("#### 📝 Executive Summary")
            st.info(f"\"{analysis['summary']}\"")
            st.caption(f"Method: {analysis['summary_details'].get('method', 'Extractive Summarization')}")

        with r2:
            st.markdown("#### 🚨 Priority Justification")
            reasons = analysis.get("priority_reasons", [])
            for r in reasons:
                st.markdown(f"• **{r}**")
            st.caption(analysis.get("priority_score_info", "Heuristic multi-signal priority calculation"))

        # Detected Entities
        st.markdown("#### 🏷️ Detected Entities")
        entities = analysis.get("entities", [])
        if entities:
            ent_badges = []
            for e in entities:
                ent_badges.append(
                    f'<span class="badge badge-entity">{e["label"]}: <strong>{e["text"]}</strong> ({e["source"].split()[0]})</span>'
                )
            st.markdown(" ".join(ent_badges), unsafe_allow_html=True)
        else:
            st.write("No named or domain-specific entities detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Resolution Workflow & Customer Response
        res1, res2 = st.columns(2)
        with res1:
            st.markdown("#### 🛠️ Recommended Action for Agent")
            st.warning(analysis["recommended_action"])
            for note in analysis.get("agent_notes", []):
                st.caption(f"📌 {note}")

        with res2:
            st.markdown("#### ✉️ Suggested Customer Response Draft")
            st.success(f"\"{analysis['suggested_response']}\"")
            st.caption("Agent policy: Templates are drafts. Never confirm payment reversal or item delivery without system check.")

        # Interactive Complaint Assistant Section
        st.markdown("---")
        st.subheader("💬 Interactive Complaint Assistant")
        st.markdown(
            "Ask local questions about this complaint analysis. Grounded deterministically without external LLM APIs."
        )

        # Quick question prompt buttons
        q_cols = st.columns(4)
        quick_questions = [
            "What is the issue?",
            "Why is the priority high?",
            "Which department should handle this?",
            "What response should we send?",
        ]
        for q_idx, q_text in enumerate(quick_questions):
            with q_cols[q_idx]:
                if st.button(q_text, key=f"qa_btn_{q_idx}", use_container_width=True):
                    st.session_state["assistant_question"] = q_text

        user_q = st.text_input(
            "Your Question:",
            value=st.session_state.get("assistant_question", ""),
            placeholder="e.g. Why is this complaint high priority?",
        )

        if user_q.strip():
            answer = assistant.answer_question(user_q, analysis)
            st.markdown(f"**Assistant:**\n{answer}")
