"""Tests for Extractive Summarization."""

from src.summarization.summarizer import ComplaintSummarizer


def test_extractive_summarization():
    summarizer = ComplaintSummarizer()
    text = (
        "My order #ORD45821 was supposed to arrive yesterday but it still hasn't arrived. "
        "I tried contacting the delivery partner but there is no response. "
        "This is extremely frustrating because I have an event tomorrow."
    )
    result = summarizer.summarize(text, max_sentences=2)

    assert "summary" in result
    assert len(result["summary"]) > 0
    assert result["summary_sentence_count"] <= 2
    assert "ORD45821" in result["summary"] or "arrive" in result["summary"]


def test_summarization_short():
    summarizer = ComplaintSummarizer()
    text = "Short complaint text."
    result = summarizer.summarize(text)
    assert result["summary"] == text
