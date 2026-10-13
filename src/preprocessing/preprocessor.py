"""Text Preprocessing module for ResolveAI.
Demonstrates:
- Lowercasing and normalization
- Sentence tokenization
- Word tokenization
- Punctuation removal
- Stopword removal
- Bag of Words & TF-IDF feature demonstration
"""

import re
import string
from typing import Dict, List, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextPreprocessor:
    """Handles text preprocessing and BoW/TF-IDF feature extraction."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> Dict[str, Any]:
        """Performs full step-by-step preprocessing of input text."""
        if not text or not text.strip():
            return {
                "original_text": text or "",
                "sentences": [],
                "tokens": [],
                "clean_tokens": [],
                "stopwords_removed": [],
                "normalized_text": "",
                "sentence_count": 0,
                "token_count": 0,
                "clean_token_count": 0,
            }

        # 1. Sentence Tokenization
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # 2. Word Tokenization (preserving original casing first)
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

        # 3. Normalization & Lowercasing
        tokens_lower = [t.lower() for t in tokens]

        # 4. Stopword and Punctuation Removal
        punct_set = set(string.punctuation) | {"''", "``", "--", "...", "’", "‘", "“", "”"}
        clean_tokens = []
        removed_stopwords = []

        for token in tokens_lower:
            if token in self.stop_words:
                removed_stopwords.append(token)
            elif token not in punct_set and re.search(r"\w", token):
                clean_tokens.append(token)

        normalized_text = " ".join(clean_tokens)

        return {
            "original_text": text,
            "sentences": sentences,
            "tokens": tokens,
            "tokens_lower": tokens_lower,
            "clean_tokens": clean_tokens,
            "stopwords_removed": list(dict.fromkeys(removed_stopwords)),
            "normalized_text": normalized_text,
            "sentence_count": len(sentences),
            "token_count": len(tokens),
            "clean_token_count": len(clean_tokens),
        }

    def compute_bag_of_words(self, text: str, max_features: int = 15) -> Dict[str, Any]:
        """Computes Bag of Words representation for a given text or corpus."""
        if not text or not text.strip():
            return {"vocabulary": {}, "feature_names": [], "counts": []}

        vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
        try:
            bow_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out().tolist()
            counts = bow_matrix.toarray()[0].tolist()
            vocab_counts = sorted(
                zip(feature_names, counts), key=lambda x: x[1], reverse=True
            )
            return {
                "vocabulary": vectorizer.vocabulary_,
                "feature_names": feature_names,
                "counts": counts,
                "term_frequency_pairs": vocab_counts,
            }
        except ValueError:
            # When text contains only stopwords
            return {"vocabulary": {}, "feature_names": [], "counts": [], "term_frequency_pairs": []}

    def compute_tfidf(self, text: str, max_features: int = 15) -> Dict[str, Any]:
        """Computes local TF-IDF score representation for inspection and feature importance."""
        if not text or not text.strip():
            return {"feature_names": [], "scores": [], "term_score_pairs": []}

        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out().tolist()
            scores = [round(float(s), 4) for s in tfidf_matrix.toarray()[0]]
            term_score_pairs = sorted(
                zip(feature_names, scores), key=lambda x: x[1], reverse=True
            )
            return {
                "feature_names": feature_names,
                "scores": scores,
                "term_score_pairs": term_score_pairs,
            }
        except ValueError:
            return {"feature_names": [], "scores": [], "term_score_pairs": []}
