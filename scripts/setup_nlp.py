"""NLP Setup Script for ResolveAI.
Downloads and verifies all necessary NLTK corpora and spaCy models.
Run this script once before starting the application:
    python scripts/setup_nlp.py
"""

import sys
import nltk


def setup_nltk():
    """Download required NLTK resources."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "vader_lexicon",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "brown",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "omw-1.4",
    ]
    print("Checking and downloading NLTK datasets...")
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"  [✓] NLTK: {resource}")
        except Exception as e:
            print(f"  [!] Warning downloading NLTK resource '{resource}': {e}")


def setup_spacy():
    """Verify or download spaCy en_core_web_sm model."""
    print("Checking spaCy model (en_core_web_sm)...")
    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            print("  [✓] spaCy model 'en_core_web_sm' is already loaded.")
        except Exception:
            print("  Downloading 'en_core_web_sm' via spacy.cli.download...")
            from spacy.cli import download

            download("en_core_web_sm")
            print("  [✓] spaCy 'en_core_web_sm' downloaded successfully.")
    except ImportError:
        print("  [X] spaCy is not installed. Please run `pip install spacy`.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("ResolveAI: Setting up local NLP resources")
    print("=" * 60)
    setup_nltk()
    setup_spacy()
    print("=" * 60)
    print("All NLP resources verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
