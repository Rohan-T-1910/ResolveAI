"""Porter Stemmer analysis module using NLTK PorterStemmer.
Demonstrates:
- Word stemming
- Comparison between original words and stemmed forms
- Detection of suffix stripping rules
"""

from typing import List, Dict, Any
from nltk.stem import PorterStemmer


class ComplaintStemmer:
    """Handles Porter Stemmer operations and comparative visual output."""

    def __init__(self):
        self.stemmer = PorterStemmer()

    def stem_words(self, words: List[str]) -> List[Dict[str, Any]]:
        """Stems a list of words and returns detailed transformation information."""
        results = []
        seen = set()

        for word in words:
            clean_word = word.strip().lower()
            if not clean_word or not clean_word.isalpha() or clean_word in seen:
                continue

            seen.add(clean_word)
            stem = self.stemmer.stem(clean_word)
            changed = clean_word != stem

            results.append({
                "original": clean_word,
                "stem": stem,
                "changed": changed,
                "rule_note": self._explain_rule(clean_word, stem) if changed else "No stem change",
            })

        return results

    def stem_text(self, text: str) -> str:
        """Returns the stemmed string representation of a sentence/text."""
        words = text.split()
        return " ".join(self.stemmer.stem(w.lower()) for w in words)

    def _explain_rule(self, word: str, stem: str) -> str:
        """Provides an explanatory note on what suffix was stripped or modified."""
        if word.endswith("ing") and stem == word[:-3]:
            return "Stripped '-ing' progressive suffix"
        if word.endswith("ed") and (stem == word[:-2] or stem == word[:-1]):
            return "Stripped '-ed' past tense suffix"
        if word.endswith("es") and stem == word[:-2]:
            return "Stripped '-es' plural/verb suffix"
        if word.endswith("s") and not word.endswith("ss") and stem == word[:-1]:
            return "Stripped '-s' suffix"
        if word.endswith("tion") and stem == word[:-4]:
            return "Stripped '-tion' noun nominalization suffix"
        if word.endswith("ly") and stem == word[:-2]:
            return "Stripped '-ly' adverb suffix"
        if word.endswith("ment") and stem == word[:-4]:
            return "Stripped '-ment' suffix"
        return "Applied Porter Stemming rule transformation"
