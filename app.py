"""ResolveAI: An NLP-Based Intelligent Customer Complaint Analysis and Resolution Assistant.
Main Streamlit application entry point.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import ResolvePipeline
from src.styles import apply_custom_styles
from pages_impl.dashboard import render_dashboard
from pages_impl.analyzer import render_analyzer
from pages_impl.history import render_history

# Page Configuration
st.set_page_config(
    page_title="ResolveAI - Intelligent Complaint Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing ResolveAI NLP Pipeline...")
def get_pipeline():
    """Initializes and caches the NLP pipeline."""
    return ResolvePipeline()


def main():
    apply_custom_styles()
    pipeline = get_pipeline()

    # Sidebar Navigation
    st.sidebar.markdown("""
        <div class="sidebar-title">🛡️ ResolveAI</div>
        <div class="sidebar-sub">NLP Complaint Resolution Assistant</div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Dashboard",
        "Complaint Analyzer",
        "Complaint History",
    ]

    # Handle navigation state changes from other pages (e.g., Load in Analyzer from History)
    if "nav_selection" in st.session_state and st.session_state["nav_selection"] in nav_options:
        default_index = nav_options.index(st.session_state["nav_selection"])
        del st.session_state["nav_selection"]
    else:
        default_index = 0

    page = st.sidebar.radio(
        "Navigation Menu:",
        nav_options,
        index=default_index,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick System Info:**")
    st.sidebar.caption("• 8 Categories (TF-IDF + LogReg)")
    st.sidebar.caption("• NLTK VADER Sentiment Analysis")
    st.sidebar.caption("• spaCy + Regex Hybrid NER")
    st.sidebar.caption("• Gensim Word2Vec & WordNet WSD")
    st.sidebar.caption("• Local Extractive Summarization")
    st.sidebar.markdown("---")
    st.sidebar.caption("ResolveAI • Customer Support Assistant")

    # Route to selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Complaint Analyzer":
        render_analyzer(pipeline)
    elif page == "Complaint History":
        render_history()


if __name__ == "__main__":
    main()
