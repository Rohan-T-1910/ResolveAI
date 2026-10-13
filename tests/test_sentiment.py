"""Tests for VADER sentiment analysis."""

from src.sentiment.analyzer import ComplaintSentimentAnalyzer


def test_negative_sentiment():
    analyzer = ComplaintSentimentAnalyzer()
    text = "This is extremely frustrating and terrible service. Nobody is answering my calls!"
    result = analyzer.analyze(text)

    assert result["sentiment"] == "Negative"
    assert result["compound_score"] < -0.2
    assert "scores" in result
    assert "disclaimer" in result


def test_positive_sentiment():
    analyzer = ComplaintSentimentAnalyzer()
    text = "Thank you so much, the customer care executive was very helpful and wonderful."
    result = analyzer.analyze(text)

    assert result["sentiment"] == "Positive"
    assert result["compound_score"] > 0.2


def test_neutral_sentiment():
    analyzer = ComplaintSentimentAnalyzer()
    text = "Inquiring about delivery schedule for order ORD123."
    result = analyzer.analyze(text)

    assert result["sentiment"] in ["Neutral", "Positive"]
