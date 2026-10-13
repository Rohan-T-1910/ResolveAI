"""Tests for Porter Stemmer analysis."""

from src.stemming.stemmer import ComplaintStemmer


def test_stemming_transformations():
    stemmer = ComplaintStemmer()
    words = ["delivered", "delivering", "delivers", "refunded", "processing"]
    results = stemmer.stem_words(words)

    stem_map = {r["original"]: r["stem"] for r in results}
    assert stem_map["delivered"] == stem_map["delivering"] == stem_map["delivers"] == "deliv"
    assert stem_map["processing"] == "process"


def test_stem_sentence():
    stemmer = ComplaintStemmer()
    text = "The courier was delivering packages"
    stemmed = stemmer.stem_text(text)
    assert "deliv" in stemmed
