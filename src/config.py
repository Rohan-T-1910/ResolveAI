"""ResolveAI configuration module.
Defines constants, paths, categories, departments, and default settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"

DATASET_PATH = DATA_DIR / "sample_complaints.csv"
DB_PATH = DATA_DIR / "resolve_ai.db"

CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
WORD2VEC_PATH = MODELS_DIR / "word2vec.model"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Complaint Categories
CATEGORIES = [
    "Delivery Issue",
    "Payment Issue",
    "Refund Issue",
    "Product Issue",
    "Account Issue",
    "Technical Issue",
    "Cancellation",
    "Other",
]

# Department Mapping
CATEGORY_DEPARTMENT_MAP = {
    "Delivery Issue": "Logistics & Delivery Operations",
    "Payment Issue": "Billing & Payments Team",
    "Refund Issue": "Finance & Accounts Receivable",
    "Product Issue": "Quality Assurance & Vendor Management",
    "Account Issue": "Customer Security & Identity Services",
    "Technical Issue": "IT Systems & Platform Engineering",
    "Cancellation": "Order Fulfillment & Retention Desk",
    "Other": "General Customer Support Services",
}

# Priority Levels
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# Sentiment Labels
SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]


