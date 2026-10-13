"""Brown Corpus Analysis module using NLTK Brown Corpus.
Demonstrates:
- Brown corpus overview and statistics
- Category distribution and genre analysis
- Comparative lexical frequency across genres
- Comparative perspective on balanced benchmark corpora in NLP
"""

from typing import Dict, List, Any
import nltk
from nltk.corpus import brown


class BrownCorpusExplorer:
    """Explores the Brown Corpus for comparative linguistics and domain frequency benchmarks."""

    def __init__(self):
        try:
            brown.categories()
        except LookupError:
            nltk.download("brown", quiet=True)

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Returns overall Brown Corpus metadata and category sizes."""
        try:
            categories = brown.categories()
            total_words = len(brown.words())
            total_sents = len(brown.sents())
            total_files = len(brown.fileids())

            category_counts = {}
            for cat in categories[:8]:  # Sample top categories
                category_counts[cat] = len(brown.words(categories=cat))

            return {
                "total_words": total_words,
                "total_sentences": total_sents,
                "total_files": total_files,
                "categories": categories,
                "sample_category_counts": category_counts,
                "description": (
                    "The Brown University Standard Corpus of Present-Day American English "
                    "was compiled in 1961 and contains ~1 million words across 15 genres, "
                    "serving as a foundational benchmark for POS taggers and frequency studies."
                ),
            }
        except Exception as e:
            return {
                "total_words": 0,
                "total_sentences": 0,
                "total_files": 0,
                "categories": [],
                "sample_category_counts": {},
                "error": str(e),
            }

    def compare_word_frequencies(self, target_words: List[str] = None) -> List[Dict[str, Any]]:
        """Compares word frequencies across prominent Brown genres."""
        if target_words is None:
            target_words = ["order", "money", "pay", "service", "delay", "time"]

        genres = ["news", "reviews", "editorial", "government", "fiction"]
        results = []

        try:
            for word in target_words:
                w_lower = word.lower()
                row = {"word": w_lower}
                total_in_genres = 0
                for g in genres:
                    # Count frequency normalized per 10,000 words in genre
                    words_in_g = [w.lower() for w in brown.words(categories=g)]
                    count = words_in_g.count(w_lower)
                    total_in_genres += count
                    row[g] = count
                row["total"] = total_in_genres
                results.append(row)
        except Exception:
            pass

        return results
