# ResolveAI

**An NLP-Based Intelligent Customer Complaint Analysis and Resolution Assistant**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-3776AB.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/NER-spaCy-09A3D5.svg)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

ResolveAI automates first-level triaging and enrichment of incoming e-commerce customer support complaints. By parsing unstructured customer emails, tickets, and chat messages, ResolveAI extracts key entities, classifies the issue, evaluates customer sentiment, identifies urgent SLA/fraud signals, generates an extractive summary, and provides agents with actionable routing recommendations and policy-safe draft responses.

---

## 🚀 Key Features

* **Multi-Class Complaint Classification**: TF-IDF + Multinomial Logistic Regression model categorizing tickets into 8 support domains (*Delivery, Payment, Refund, Product, Account, Technical, Cancellation, Other*).
* **Sentiment & Valence Scoring**: NLTK VADER sentiment engine evaluating emotional polarity (-1.0 to +1.0) and frustration intensity.
* **Hybrid Named Entity Recognition (NER)**: Combines statistical spaCy models with targeted regex patterns to extract Order IDs (`#ORD...`), monetary amounts (`₹...`), emails, phone numbers, and dates.
* **Multi-Signal Urgency & Priority Triaging**: Heuristic engine evaluating sentiment severity, high transaction amounts, and fraud/legal keywords to determine urgency (*Low, Medium, High, Critical*) with auditable justifications.
* **Extractive Summarization**: Local sentence-scoring summarizer highlighting the core issue without external LLM dependencies.
* **Automated Resolution Routing & Safe Drafts**: Maps tickets to departments (*Billing, Logistics, Quality Assurance, etc.*), recommends operational agent workflows, and drafts policy-compliant customer replies.
* **Persistent History & Analytics**: SQLite database storing full diagnostic records with an interactive dashboard displaying real-time KPIs and ticket distribution trends.

---

## 🏗️ Architecture

```text
Raw Customer Ticket
        │
        ├─► [Text Preprocessing] ──► Tokenization, Normalization, Stopword Removal, Stemming
        │
        ├─► [Feature Extraction] ──► TF-IDF N-grams (1, 2)
        │         │
        │         └─► [ML Classifier] ──► 8 Support Categories (Logistic Regression)
        │
        ├─► [Sentiment Engine]   ──► NLTK VADER Compound Polarity Score
        │
        ├─► [Hybrid NER]         ──► spaCy Entities + Domain Regex (Order IDs, ₹ Amounts, Contacts)
        │
        ├─► [Summarizer]         ──► Salience-Based Extractive Sentence Scoring
        │
        └─► [Priority Engine]    ──► Multi-Signal Urgency Assignment (SLA & Fraud Rules)
                  │
                  ▼
   [Resolution & Response Engine] ──► Department Routing + Policy-Safe Draft Reply
                  │
                  ├─► SQLite Storage (`data/resolve_ai.db`)
                  └─► Streamlit UI Dashboard & Agent Workplace
```

---

## 📂 Repository Structure

```text
ResolveAI/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── sample_complaints.csv   # Labeled complaint dataset (8 categories)
│   └── resolve_ai.db           # Persistent SQLite database
│
├── models/
│   ├── classifier.joblib       # Trained Logistic Regression model
│   ├── tfidf_vectorizer.joblib # Fitted TF-IDF vectorizer
│   ├── word2vec.model          # Gensim Word2Vec skip-gram embeddings
│   └── evaluation_metrics.json # Model performance metrics & confusion matrix
│
├── pages_impl/                 # Streamlit UI views
│   ├── dashboard.py            # Analytics KPIs and distribution charts
│   ├── analyzer.py             # Single complaint analyzer & Q&A assistant
│   └── history.py              # Filterable complaint logs and search
│
├── src/                        # Core NLP & ML modules
│   ├── pipeline.py             # Master pipeline coordinator
│   ├── classification/         # Classifier training and inference
│   ├── sentiment/              # VADER sentiment analysis
│   ├── priority/               # Urgency detection & heuristic rules
│   ├── ner/                    # Hybrid spaCy + Regex entity extraction
│   ├── summarization/          # Extractive summarizer
│   ├── resolution/             # Department routing & response drafts
│   ├── database/               # SQLite CRUD operations
│   └── preprocessing/          # Tokenization, cleaning & TF-IDF
│
├── scripts/
│   ├── setup_nlp.py            # NLTK data & spaCy model downloader
│   ├── train_model.py          # Model training pipeline
│   └── seed_db.py              # Database seeding script
│
└── tests/                      # Automated test suite (28 pytest unit tests)
```

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Rohan-T-1910/ResolveAI.git
cd ResolveAI
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP Assets
```bash
python scripts/setup_nlp.py
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` (or `8502`) in your browser.

---

## 🧪 Testing

Execute the comprehensive test suite:
```bash
pytest -v
```
All 28 unit and end-to-end integration tests validate data preprocessing, classification accuracy, entity extraction, sentiment scoring, and database integrity.

---

## 💻 Tech Stack

* **Language**: Python 3.11+
* **Frontend**: Streamlit, Plotly
* **Machine Learning**: Scikit-learn (TF-IDF, Logistic Regression)
* **NLP & Linguistics**: NLTK, spaCy (`en_core_web_sm`), Gensim (Word2Vec)
* **Database**: SQLite3
* **Testing**: pytest

---

## 📄 License

Distributed under the [MIT License](LICENSE).
