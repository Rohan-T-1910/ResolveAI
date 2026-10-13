"""Part-of-Speech Tagging module using NLTK.
Demonstrates:
- Penn Treebank POS tagging
- Human-readable grammatical role descriptions
- Tag frequency distribution and syntactic breakdown
"""

from typing import List, Dict, Tuple, Any
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

PENN_TREEBANK_TAGS = {
    "CC": "Coordinating conjunction (and, but, or)",
    "CD": "Cardinal number (one, 1, 2,000)",
    "DT": "Determiner (the, a, this, an)",
    "EX": "Existential there (there was)",
    "FW": "Foreign word",
    "IN": "Preposition / subordinating conjunction (in, of, on)",
    "JJ": "Adjective (damaged, defective, broken)",
    "JJR": "Adjective, comparative (faster, slower)",
    "JJS": "Adjective, superlative (fastest, worst)",
    "LS": "List item marker",
    "MD": "Modal (can, could, should, will)",
    "NN": "Noun, singular or mass (order, delivery, refund)",
    "NNS": "Noun, plural (orders, packages, items)",
    "NNP": "Proper noun, singular (ResolveAI, Rahul, Delhi)",
    "NNPS": "Proper noun, plural (Mondays)",
    "PDT": "Predeterminer",
    "POS": "Possessive ending ('s)",
    "PRP": "Personal pronoun (I, he, she, it, they)",
    "PRP$": "Possessive pronoun (my, your, his, her, its)",
    "RB": "Adverb (extremely, yesterday, not)",
    "RBR": "Adverb, comparative (better, later)",
    "RBS": "Adverb, superlative (best, earliest)",
    "RP": "Particle (about, off, up)",
    "TO": "to (to go, to bank)",
    "UH": "Interjection (ah, oops)",
    "VB": "Verb, base form (deliver, cancel, refund)",
    "VBD": "Verb, past tense (delivered, cancelled, arrived)",
    "VBG": "Verb, gerund / present participle (waiting, contacting)",
    "VBN": "Verb, past participle (received, delayed, broken)",
    "VBP": "Verb, non-3rd person singular present (want, need)",
    "VBZ": "Verb, 3rd person singular present (wants, needs, is)",
    "WDT": "Wh-determiner (which, that)",
    "WP": "Wh-pronoun (what, who)",
    "WP$": "Possessive wh-pronoun (whose)",
    "WRB": "Wh-adverb (where, when, why)",
    ".": "Sentence-final punctuation",
    ",": "Comma",
    ":": "Colon or ellipsis",
    "$": "Dollar or currency sign",
}


class ComplaintPOSTagger:
    """Performs Part-of-Speech tagging with detailed descriptions."""

    def __init__(self):
        try:
            pos_tag(["test"])
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    def tag_text(self, text: str) -> List[Dict[str, Any]]:
        """Tags words in text and returns token, tag, and explanation."""
        if not text or not text.strip():
            return []

        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        tagged: List[Tuple[str, str]] = pos_tag(tokens)

        results = []
        for word, tag in tagged:
            description = PENN_TREEBANK_TAGS.get(
                tag, f"Syntactic category ({tag})"
            )
            category = self._broad_category(tag)
            results.append({
                "word": word,
                "tag": tag,
                "description": description,
                "category": category,
            })

        return results

    def get_pos_summary(self, tagged_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculates frequency breakdown by broad grammatical categories."""
        counts = {
            "Nouns": 0,
            "Verbs": 0,
            "Adjectives": 0,
            "Adverbs": 0,
            "Pronouns": 0,
            "Prepositions/Conjunctions": 0,
            "Determiners": 0,
            "Other/Punctuation": 0,
        }

        for item in tagged_results:
            cat = item.get("category", "Other/Punctuation")
            if cat in counts:
                counts[cat] += 1
            else:
                counts["Other/Punctuation"] += 1

        return counts

    def _broad_category(self, tag: str) -> str:
        """Groups fine-grained Penn Treebank tags into broad categories."""
        if tag.startswith("NN"):
            return "Nouns"
        elif tag.startswith("VB"):
            return "Verbs"
        elif tag.startswith("JJ"):
            return "Adjectives"
        elif tag.startswith("RB"):
            return "Adverbs"
        elif tag.startswith("PRP"):
            return "Pronouns"
        elif tag in ("IN", "CC"):
            return "Prepositions/Conjunctions"
        elif tag in ("DT", "PDT", "WDT"):
            return "Determiners"
        else:
            return "Other/Punctuation"
