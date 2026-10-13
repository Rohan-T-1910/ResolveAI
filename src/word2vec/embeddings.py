"""Word2Vec Semantic Embedding module using Gensim.
Demonstrates:
- Local domain Word2Vec continuous vector space representation
- Word vector preview and dimensional properties
- Top-N semantic nearest neighbors
- Word-to-word cosine similarity calculation
- Safe handling of out-of-vocabulary (OOV) tokens
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from gensim.models import Word2Vec
from src.config import WORD2VEC_PATH, DATASET_PATH
import pandas as pd
from nltk.tokenize import word_tokenize


class ComplaintWord2Vec:
    """Manages training and querying of a local Word2Vec model on complaint texts."""

    def __init__(self, model_path: Path = WORD2VEC_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[Word2Vec] = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Loads saved Word2Vec model if exists, or trains a fresh one."""
        if self.model_path.exists():
            try:
                self.model = Word2Vec.load(str(self.model_path))
                return
            except Exception:
                pass
        self.train_and_save()

    def train_and_save(self, csv_path: Path = DATASET_PATH, vector_size: int = 50, window: int = 4, min_count: int = 1):
        """Trains a Word2Vec model on the complaints dataset."""
        sentences = []
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            for text in df["text"].dropna():
                tokens = [w.lower() for w in word_tokenize(str(text)) if w.isalpha()]
                if tokens:
                    sentences.append(tokens)

        if not sentences:
            # Fallback baseline corpus if dataset is missing
            sentences = [
                ["order", "delayed", "delivery", "courier", "package", "arrived", "shipping"],
                ["refund", "amount", "money", "bank", "account", "transaction", "payment"],
                ["damaged", "broken", "defective", "product", "quality", "item", "replacement"],
                ["cancel", "cancellation", "order", "status", "stop", "dispatch"],
                ["account", "locked", "password", "login", "reset", "email", "phone", "otp"],
            ]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,  # Skip-gram
            epochs=25,
            seed=42,
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_path))

    def get_word_vector(self, word: str) -> Dict[str, Any]:
        """Returns vector properties and numerical values for a word."""
        w_lower = (word or "").strip().lower()
        if not self.model or w_lower not in self.model.wv:
            return {
                "word": w_lower,
                "in_vocabulary": False,
                "vector_size": self.model.vector_size if self.model else 0,
                "vector_preview": [],
                "message": f"Word '{w_lower}' is out-of-vocabulary (OOV) for this local complaint model.",
            }

        vec = self.model.wv[w_lower]
        return {
            "word": w_lower,
            "in_vocabulary": True,
            "vector_size": len(vec),
            "vector_preview": [round(float(v), 4) for v in vec[:10]],
            "full_vector": [float(v) for v in vec],
            "norm": round(float(np.linalg.norm(vec)), 4),
            "message": f"Vector representation found ({len(vec)} dimensions).",
        }

    def get_similar_words(self, word: str, topn: int = 6) -> Dict[str, Any]:
        """Finds the most semantically similar words in vector space."""
        w_lower = (word or "").strip().lower()
        if not self.model or w_lower not in self.model.wv:
            return {
                "word": w_lower,
                "in_vocabulary": False,
                "similar_words": [],
                "message": f"Word '{w_lower}' is not in the vocabulary.",
            }

        try:
            results = self.model.wv.most_similar(w_lower, topn=topn)
            similar = [{"word": w, "similarity": round(float(score), 4)} for w, score in results]
            return {
                "word": w_lower,
                "in_vocabulary": True,
                "similar_words": similar,
                "message": f"Found {len(similar)} nearest semantic neighbors.",
            }
        except Exception as e:
            return {
                "word": w_lower,
                "in_vocabulary": True,
                "similar_words": [],
                "message": f"Error computing similarity: {e}",
            }

    def compute_similarity(self, word1: str, word2: str) -> Dict[str, Any]:
        """Computes cosine similarity between two words."""
        w1 = (word1 or "").strip().lower()
        w2 = (word2 or "").strip().lower()

        if not self.model:
            return {"word1": w1, "word2": w2, "similarity": None, "message": "Model not loaded."}

        w1_in = w1 in self.model.wv
        w2_in = w2 in self.model.wv

        if not w1_in or not w2_in:
            missing = []
            if not w1_in:
                missing.append(f"'{w1}'")
            if not w2_in:
                missing.append(f"'{w2}'")
            return {
                "word1": w1,
                "word2": w2,
                "similarity": None,
                "message": f"Out-of-vocabulary: {', '.join(missing)} not found in local model.",
            }

        sim = float(self.model.wv.similarity(w1, w2))
        return {
            "word1": w1,
            "word2": w2,
            "similarity": round(sim, 4),
            "percentage": round(sim * 100, 2),
            "message": f"Cosine similarity between '{w1}' and '{w2}' is {round(sim, 4)}.",
        }

    def get_vocabulary_sample(self, limit: int = 30) -> List[str]:
        """Returns sample vocabulary words present in the model."""
        if not self.model:
            return []
        return list(self.model.wv.index_to_key[:limit])
