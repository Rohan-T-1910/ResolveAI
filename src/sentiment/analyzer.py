"""Sentiment Analysis module using NLTK VADER.
Demonstrates:
- Rule-based valence-aware sentiment analysis
- Compound score (-1.0 to +1.0) and component breakdown (pos, neu, neg)
- Transparent classification thresholding
- Explicit analytical disclaimer on model outputs vs psychological states
"""

from typing import Dict, Any, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class ComplaintSentimentAnalyzer:
    """Analyzes complaint sentiment using the NLTK VADER lexicon."""

    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, Any]:
        """Calculates compound, positive, neutral, negative scores and label."""
        clean_text = (text or "").strip()
        if not clean_text:
            return {
                "sentiment": "Neutral",
                "compound_score": 0.0,
                "scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0},
                "explanation": "No text provided for sentiment analysis.",
                "disclaimer": self._get_disclaimer(),
            }

        scores = self.sia.polarity_scores(clean_text)
        compound = round(float(scores["compound"]), 4)

        if compound >= 0.05:
            sentiment = "Positive"
            explanation = f"Positive tone detected (compound score: {compound:+.2f})."
        elif compound <= -0.05:
            sentiment = "Negative"
            explanation = f"Negative or dissatisfied tone detected (compound score: {compound:+.2f})."
        else:
            sentiment = "Neutral"
            explanation = f"Neutral tone detected (compound score: {compound:+.2f})."

        return {
            "sentiment": sentiment,
            "compound_score": compound,
            "scores": {
                "pos": round(float(scores["pos"]), 3),
                "neu": round(float(scores["neu"]), 3),
                "neg": round(float(scores["neg"]), 3),
                "compound": compound,
            },
            "explanation": explanation,
            "disclaimer": self._get_disclaimer(),
        }

    def _get_disclaimer(self) -> str:
        return (
            "Notice: The sentiment score is a statistical lexical model output (NLTK VADER) "
            "evaluating token polarity and valence; it does not measure subjective psychological states."
        )
