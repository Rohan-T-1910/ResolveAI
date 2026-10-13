"""Chunking and Chunk Tree module using NLTK.
Demonstrates:
- RegexpParser shallow parsing
- Noun Phrase (NP), Verb Phrase (VP), and Prepositional Phrase (PP) chunking
- Chunk tree generation and textual hierarchy representation
"""

from typing import List, Dict, Any, Tuple
import nltk
from nltk import pos_tag, RegexpParser, Tree
from nltk.tokenize import word_tokenize

# Grammars for chunking e-commerce complaints
CHUNK_GRAMMAR = r"""
    NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}    # Noun Phrase (e.g., 'the delivery partner', 'my order')
    VP: {<MD>?<VB.*>+(<RB.*>)*<VB.*>*}  # Verb Phrase (e.g., 'was supposed to arrive', 'cannot log in')
    PP: {<IN><DT|PRP\$>?<JJ.*>*<NN.*>+} # Prepositional Phrase (e.g., 'with the delivery partner')
"""


class ComplaintChunker:
    """Performs syntactic chunking and parses text into chunk trees."""

    def __init__(self):
        self.parser = RegexpParser(CHUNK_GRAMMAR)

    def chunk_text(self, text: str) -> Dict[str, Any]:
        """Parses text into chunks, returns tree structure and extracted phrases."""
        if not text or not text.strip():
            return {
                "tree": None,
                "tree_string": "",
                "chunks": [],
                "noun_phrases": [],
                "verb_phrases": [],
                "bracketed_text": "",
            }

        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        try:
            tagged = pos_tag(tokens)
        except Exception:
            tagged = [(t, "NN") for t in tokens]

        tree: Tree = self.parser.parse(tagged)

        chunks = []
        noun_phrases = []
        verb_phrases = []

        for subtree in tree:
            if isinstance(subtree, Tree):
                label = subtree.label()
                words = [w for w, t in subtree.leaves()]
                phrase_text = " ".join(words)
                tag_details = [(w, t) for w, t in subtree.leaves()]

                chunk_entry = {
                    "type": label,
                    "phrase": phrase_text,
                    "leaves": tag_details,
                }
                chunks.append(chunk_entry)

                if label == "NP":
                    noun_phrases.append(phrase_text)
                elif label == "VP":
                    verb_phrases.append(phrase_text)

        # Build readable bracketed string representation
        bracketed_tokens = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                phrase = " ".join(w for w, _ in subtree.leaves())
                bracketed_tokens.append(f"[{subtree.label()}: {phrase}]")
            else:
                bracketed_tokens.append(subtree[0])
        bracketed_text = " ".join(bracketed_tokens)

        # Build readable multi-line indented tree string
        tree_string = self._format_tree(tree)

        return {
            "tree": tree,
            "tree_string": tree_string,
            "chunks": chunks,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "bracketed_text": bracketed_text,
        }

    def _format_tree(self, tree: Tree, indent: int = 0) -> str:
        """Formats an NLTK Tree into a clean, ASCII tree string."""
        lines = []
        prefix = "  " * indent
        if isinstance(tree, Tree):
            lines.append(f"{prefix}└── [{tree.label()}]")
            for child in tree:
                lines.append(self._format_tree(child, indent + 1))
        else:
            word, tag = tree
            lines.append(f"{prefix}    ├── {word} / {tag}")
        return "\n".join(lines)
