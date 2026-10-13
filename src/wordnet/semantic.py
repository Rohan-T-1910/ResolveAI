"""WordNet Semantic Analysis module using NLTK WordNet.
Demonstrates:
- Synsets and definitions
- Synonyms (lemmas)
- Antonyms
- Hypernyms (superordinates) and Hyponyms (subordinates)
- Graceful error handling for missing/OOV terms
"""

from typing import Dict, List, Any
import nltk
from nltk.corpus import wordnet as wn


class WordNetAnalyzer:
    """Explores semantic relationships using the Princeton WordNet lexical database."""

    def __init__(self):
        try:
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """Looks up a word in WordNet and extracts semantic relations."""
        clean_word = (word or "").strip().lower()
        if not clean_word:
            return {
                "word": "",
                "found": False,
                "message": "Please provide a valid word to analyze.",
                "synsets": [],
                "synonyms": [],
                "antonyms": [],
                "definitions": [],
            }

        synsets = wn.synsets(clean_word)
        if not synsets:
            return {
                "word": clean_word,
                "found": False,
                "message": f"No WordNet synset entries found for '{clean_word}'.",
                "synsets": [],
                "synonyms": [],
                "antonyms": [],
                "definitions": [],
            }

        synonyms = set()
        antonyms = set()
        definitions = []
        synset_details = []

        for s in synsets:
            # Synonyms & Antonyms from lemmas
            for lemma in s.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name.lower() != clean_word:
                    synonyms.add(lemma_name)
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace("_", " "))

            # Definitions and examples
            definitions.append(f"[{s.pos()}] {s.definition()}")

            # Hypernyms and Hyponyms
            hypernyms = [h.name().split(".")[0].replace("_", " ") for h in s.hypernyms()[:4]]
            hyponyms = [h.name().split(".")[0].replace("_", " ") for h in s.hyponyms()[:4]]

            synset_details.append({
                "synset_name": s.name(),
                "pos": s.pos(),
                "definition": s.definition(),
                "examples": s.examples()[:2],
                "hypernyms": hypernyms,
                "hyponyms": hyponyms,
            })

        return {
            "word": clean_word,
            "found": True,
            "total_synsets": len(synsets),
            "synonyms": sorted(list(synonyms))[:12],
            "antonyms": sorted(list(antonyms))[:12],
            "definitions": definitions[:6],
            "synset_details": synset_details[:5],
            "message": f"Found {len(synsets)} sense(s) in WordNet.",
        }

    def get_related_complaint_words(self, words: List[str]) -> List[Dict[str, Any]]:
        """Quick semantic preview for multiple words in a complaint."""
        results = []
        for w in words:
            if len(w) > 2 and w.isalpha():
                info = self.analyze_word(w)
                if info["found"]:
                    results.append({
                        "word": w,
                        "top_synonyms": info["synonyms"][:4],
                        "definition": info["definitions"][0] if info["definitions"] else "",
                    })
        return results
