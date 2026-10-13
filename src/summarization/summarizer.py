"""Local Extractive Text Summarization module.
Demonstrates:
- Extractive sentence-scoring summarization without external LLMs
- Sentence importance calculation using term frequency, domain keywords, and positional weighting
- Generation of concise, actionable summaries for customer support agents
"""

import re
from typing import Dict, Any, List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class ComplaintSummarizer:
    """Performs local extractive summarization on complaint texts."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

        # Domain salience keywords
        self.salience_keywords = {
            "order", "delivery", "refund", "charged", "arrived", "damaged", "broken",
            "cancelled", "account", "locked", "payment", "courier", "partner", "failed",
            "defective", "unauthorized", "tracking", "error", "credited", "waiting"
        }

    def summarize(self, text: str, max_sentences: int = 2) -> Dict[str, Any]:
        """Extracts the most salient sentences to formulate an executive summary."""
        clean_text = (text or "").strip()
        if not clean_text:
            return {
                "summary": "No complaint text provided to summarize.",
                "method": "Extractive Summarization",
                "sentence_count": 0,
                "selected_sentences": [],
            }

        try:
            sentences = sent_tokenize(clean_text)
        except Exception:
            sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]

        if len(sentences) <= max_sentences:
            return {
                "summary": clean_text,
                "method": "Direct Extraction (Concise Original)",
                "sentence_count": len(sentences),
                "selected_sentences": sentences,
            }

        # Calculate word frequency table
        words = [w.lower() for w in word_tokenize(clean_text) if w.isalnum() and w.lower() not in self.stop_words]
        if not words:
            return {
                "summary": sentences[0],
                "method": "First Sentence Heuristic",
                "sentence_count": len(sentences),
                "selected_sentences": [sentences[0]],
            }

        freq_table = {}
        for w in words:
            freq_table[w] = freq_table.get(w, 0) + 1

        # Normalize frequencies
        max_freq = max(freq_table.values()) if freq_table else 1
        for w in freq_table:
            freq_table[w] = freq_table[w] / max_freq

        # Score sentences
        sentence_scores = []
        for idx, sentence in enumerate(sentences):
            sent_words = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
            if not sent_words:
                continue

            score = 0.0
            for w in sent_words:
                if w in freq_table:
                    score += freq_table[w]
                if w in self.salience_keywords:
                    score += 1.5

            # Position weight: first and last sentences often contain core complaint and demand
            if idx == 0:
                score *= 1.3
            elif idx == len(sentences) - 1:
                score *= 1.2

            # Length normalization (avoid very short fragments or run-on sentences)
            norm_score = score / (len(sent_words) ** 0.5)
            sentence_scores.append((idx, sentence, norm_score))

        # Select top N sentences based on score, but preserve chronological order
        top_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:max_sentences]
        ordered_sentences = sorted(top_sentences, key=lambda x: x[0])

        summary_text = " ".join(s[1].strip() for s in ordered_sentences)

        return {
            "summary": summary_text,
            "method": "Local Extractive Salience Scoring",
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(ordered_sentences),
            "selected_sentences": [s[1] for s in ordered_sentences],
        }
