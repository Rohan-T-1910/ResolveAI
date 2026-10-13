"""About View for ResolveAI."""

import streamlit as st


def render_about():
    """Renders the About ResolveAI architecture and system overview page."""
    st.title("ℹ️ About ResolveAI")
    st.markdown(
        "**ResolveAI: An NLP-Based Intelligent Customer Complaint Analysis and Resolution Assistant**"
    )

    st.markdown("""
    ### 🎯 Problem Statement & Objective
    Customer support desks receive hundreds of incoming inquiries and complaints daily across chat, email, and tickets.
    Manual triaging is slow, error-prone, and causes SLA breaches for high-impact tickets (such as fraudulent charges or delayed deliveries).
    
    **ResolveAI** automates first-level triaging and ticket enrichment:
    - **Automated Categorization**: Multi-class ML classification into 8 support categories
    - **Sentiment Polarity Scoring**: Real-time evaluation of customer frustration intensity
    - **Urgency Triaging**: Rule-based priority engine detecting SLA breaches and fraud signals
    - **Entity Extraction**: Hybrid spaCy NER + regex extracting Order IDs, monetary amounts, and contacts
    - **Extractive Summarization**: Highlighting the core issue for quick agent review
    - **Resolution Routing**: Instant department assignment and policy-safe draft responses
    """)

    st.markdown("---")
    st.subheader("🧩 Core NLP Architecture & Pipeline Modules")

    components = [
        ("Text Preprocessing & Normalization", "Sentence & word tokenization, case normalization, and stopword elimination in `src/preprocessing/`."),
        ("Morphological Stemming", "NLTK PorterStemmer analyzing morphological suffixes in `src/stemming/`."),
        ("Feature Extraction", "N-gram Term Frequency-Inverse Document Frequency (TF-IDF) in `src/preprocessing/`."),
        ("Syntactic POS Tagging", "Penn Treebank POS tagger classifying grammatical roles in `src/pos/`."),
        ("Shallow Parsing & Chunking", "Regular expression chunker parsing Noun Phrases (NP) and Verb Phrases (VP) in `src/chunking/`."),
        ("Hybrid Named Entity Recognition", "Statistical spaCy NER combined with domain-specific regex extractors for ORDER_ID, AMOUNT, EMAIL, and PHONE in `src/ner/`."),
        ("Lexical Semantics & Embeddings", "WordNet synset lookup and Gensim Word2Vec continuous vector embeddings in `src/wordnet/` and `src/word2vec/`."),
        ("Supervised ML Classification", "Trained TF-IDF + Logistic Regression model predicting 8 support categories with probability outputs in `src/classification/`."),
        ("Sentiment & Priority Engine", "NLTK VADER sentiment scoring combined with multi-signal heuristics for SLA routing in `src/priority/`."),
        ("Local Extractive Summarizer", "Salience-weighted sentence extraction producing concise agent summaries in `src/summarization/`."),
    ]

    for title, desc in components:
        with st.expander(f"**{title}**"):
            st.write(desc)

    st.markdown("---")
    st.subheader("📐 System Architecture")
    st.code("""
  Raw Complaint Text
        │
  Sentence & Word Tokenization  ──>  Stopword Removal & Porter Stemmer
        │
  TF-IDF Feature Extraction     ──>  Logistic Regression Classification (8 Categories)
        │
  VADER Sentiment Analysis      ──>  Valence Scoring (-1.0 to +1.0)
        │
  POS Tagging & Regexp Chunking ──>  Penn Treebank Tags & Shallow Parse Trees
        │
  spaCy NER + Regex Extractor   ──>  ORDER_ID, AMOUNT, EMAIL, PHONE, DATES
        │
  Extractive Summarizer         ──>  Salience-Scored Core Sentences
        │
  Rule-Based Priority Engine    ──>  Low / Medium / High / Critical + Reasons
        │
  Resolution Recommendation     ──>  Department Routing, Agent Actions, Response Draft
        │
  SQLite Database Storage       ──>  Persistence & Real-Time Dashboard
    """, language="text")

    st.markdown("---")
    st.subheader("⚙️ System Guardrails & Policies")
    st.markdown("""
    1. **Deterministic Response Guardrails**: Suggested responses are policy drafts and strictly phrased never to claim a transactional action has executed without human agent verification.
    2. **Local & Privacy-Preserving**: Runs completely offline without sending customer ticket contents to third-party generative APIs.
    3. **Persistent SQLite Store**: Keeps complaint histories and aggregate KPIs accessible for real-time reporting.
    """)
