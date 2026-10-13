"""Word Sense Disambiguation (WSD) module using WordNet and the Lesk algorithm.
Demonstrates:
- Context-based sense selection
- Candidate synset evaluation
- Lexical overlap counting between context and synset glosses
- Transparent confidence reporting without fabricating certainty
"""

import re
from typing import Dict, List, Any
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


class WordSenseDisambiguator:
    """Disambiguates words in context using the simplified Lesk algorithm."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def disambiguate(self, word: str, context_sentence: str) -> Dict[str, Any]:
        """Disambiguates target word within a given context sentence."""
        clean_word = (word or "").strip().lower()
        if not clean_word:
            return {
                "word": "",
                "status": "error",
                "selected_sense": None,
                "confidence": "Low",
                "explanation": "No target word provided.",
                "candidate_senses": [],
            }

        synsets = wn.synsets(clean_word)
        if not synsets:
            return {
                "word": clean_word,
                "status": "not_found",
                "selected_sense": None,
                "confidence": "None",
                "explanation": f"Word '{clean_word}' not found in WordNet synsets.",
                "candidate_senses": [],
            }

        # Tokenize and clean context words (excluding target word itself and stopwords)
        context_words = set(re.findall(r"\b\w+\b", context_sentence.lower()))
        context_words = {w for w in context_words if w not in self.stop_words and w != clean_word and len(w) > 2}

        candidate_senses = []
        best_sense = None
        max_overlap = -1
        best_overlap_words = []

        for s in synsets:
            # Gloss and examples combined
            gloss_text = s.definition() + " " + " ".join(s.examples())
            gloss_words = set(re.findall(r"\b\w+\b", gloss_text.lower()))
            gloss_words = {w for w in gloss_words if w not in self.stop_words and len(w) > 2}

            overlap = context_words.intersection(gloss_words)
            overlap_count = len(overlap)

            candidate_info = {
                "synset_name": s.name(),
                "pos": s.pos(),
                "definition": s.definition(),
                "overlap_count": overlap_count,
                "overlap_words": list(overlap),
            }
            candidate_senses.append(candidate_info)

            if overlap_count > max_overlap:
                max_overlap = overlap_count
                best_sense = candidate_info
                best_overlap_words = list(overlap)

        # Confidence assessment
        if max_overlap >= 2:
            confidence = "High"
            explanation = (
                f"Selected sense '{best_sense['synset_name']}' based on multiple shared context words: "
                f"{', '.join(best_overlap_words)}."
            )
        elif max_overlap == 1:
            confidence = "Moderate"
            explanation = (
                f"Selected sense '{best_sense['synset_name']}' based on shared context word: "
                f"'{best_overlap_words[0]}'."
            )
        else:
            # No context overlap: return first dominant sense with explicit uncertainty disclaimer
            best_sense = candidate_senses[0] if candidate_senses else None
            confidence = "Low / Uncertain"
            explanation = "Unable to confidently determine the sense due to zero contextual overlap with WordNet glosses."

        return {
            "word": clean_word,
            "status": "success",
            "context": context_sentence,
            "selected_sense": best_sense,
            "confidence": confidence,
            "max_overlap": max(0, max_overlap),
            "explanation": explanation,
            "candidate_senses": candidate_senses[:6],
            "total_candidates": len(candidate_senses),
        }
