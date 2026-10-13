"""Tests for text preprocessing and BoW/TF-IDF feature extraction."""

from src.preprocessing.preprocessor import TextPreprocessor


def test_preprocessing_tokenization():
    prep = TextPreprocessor()
    text = "My order #ORD12345 was delayed! I called support, but no response."
    result = prep.preprocess(text)

    assert result["sentence_count"] == 2
    assert result["token_count"] > 5
    assert "order" in result["clean_tokens"]
    assert "delayed" in result["clean_tokens"]
    assert len(result["clean_tokens"]) < result["token_count"]


def test_preprocessing_stopwords():
    prep = TextPreprocessor()
    text = "This is a very simple test of the stopword removal algorithm."
    result = prep.preprocess(text)

    assert "this" in result["stopwords_removed"] or "is" in result["stopwords_removed"]
    assert "test" in result["clean_tokens"]
    assert "algorithm" in result["clean_tokens"]


def test_preprocessing_empty():
    prep = TextPreprocessor()
    result = prep.preprocess("")
    assert result["sentence_count"] == 0
    assert result["token_count"] == 0


def test_bag_of_words():
    prep = TextPreprocessor()
    text = "refund refund payment order refund"
    bow = prep.compute_bag_of_words(text)

    assert "vocabulary" in bow
    assert "refund" in bow["vocabulary"]
    assert any(term == "refund" and count == 3 for term, count in bow["term_frequency_pairs"])


def test_tfidf():
    prep = TextPreprocessor()
    text = "order delayed delivery status courier"
    tfidf = prep.compute_tfidf(text)

    assert len(tfidf["feature_names"]) > 0
    assert len(tfidf["scores"]) == len(tfidf["feature_names"])
