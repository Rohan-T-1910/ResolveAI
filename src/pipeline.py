"""Unified NLP Complaint Analysis Pipeline for ResolveAI.
Executes the sequential end-to-end NLP workflow:
Raw Text
  ↓
Sentence Tokenization
  ↓
Word Tokenization
  ↓
Text Normalization
  ↓
Stopword Removal
  ↓
Stemming
  ↓
Feature Extraction
  ↓
Classification
  ↓
Sentiment Analysis
  ↓
POS Tagging
  ↓
Chunking
  ↓
Named Entity Recognition
  ↓
Semantic Analysis (WordNet & Word2Vec)
  ↓
Summarization
  ↓
Priority Detection
  ↓
Resolution Recommendation
"""

from typing import Dict, Any, Optional
from src.preprocessing.preprocessor import TextPreprocessor
from src.stemming.stemmer import ComplaintStemmer
from src.pos.tagger import ComplaintPOSTagger
from src.chunking.chunker import ComplaintChunker
from src.ner.extractor import EntityExtractor
from src.wordnet.semantic import WordNetAnalyzer
from src.word2vec.embeddings import ComplaintWord2Vec
from src.wsd.disambiguator import WordSenseDisambiguator
from src.classification.classifier import ComplaintClassifier
from src.sentiment.analyzer import ComplaintSentimentAnalyzer
from src.priority.engine import PriorityEngine
from src.summarization.summarizer import ComplaintSummarizer
from src.resolution.engine import ResolutionEngine


class ResolvePipeline:
    """Coordinates all modular NLP components into an integrated analysis workflow."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.stemmer = ComplaintStemmer()
        self.pos_tagger = ComplaintPOSTagger()
        self.chunker = ComplaintChunker()
        self.ner_extractor = EntityExtractor()
        self.wordnet_analyzer = WordNetAnalyzer()
        self.word2vec = ComplaintWord2Vec()
        self.wsd = WordSenseDisambiguator()
        self.classifier = ComplaintClassifier()
        self.sentiment_analyzer = ComplaintSentimentAnalyzer()
        self.priority_engine = PriorityEngine()
        self.summarizer = ComplaintSummarizer()
        self.resolution_engine = ResolutionEngine()

    def analyze_complaint(self, text: str, complaint_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs the entire NLP pipeline on the complaint text and returns complete analysis."""
        clean_text = (text or "").strip()

        # 1. Preprocessing (Tokenization, Normalization, Stopwords)
        prep = self.preprocessor.preprocess(clean_text)

        # 2. Stemming
        stemmed_tokens = self.stemmer.stem_words(prep["clean_tokens"])

        # 3. Bag of Words & TF-IDF Demonstration
        bow_data = self.preprocessor.compute_bag_of_words(clean_text)
        tfidf_data = self.preprocessor.compute_tfidf(clean_text)

        # 4. Classification
        classification_result = self.classifier.predict(clean_text)
        category = classification_result["category"]

        # 5. Sentiment Analysis
        sentiment_result = self.sentiment_analyzer.analyze(clean_text)
        sentiment = sentiment_result["sentiment"]
        compound_score = sentiment_result["compound_score"]

        # 6. POS Tagging
        pos_results = self.pos_tagger.tag_text(clean_text)
        pos_summary = self.pos_tagger.get_pos_summary(pos_results)

        # 7. Chunking
        chunk_data = self.chunker.chunk_text(clean_text)

        # 8. Named Entity Recognition (Standard + Domain)
        ner_data = self.ner_extractor.extract_all(clean_text)

        # 9. Summarization
        summary_data = self.summarizer.summarize(clean_text)
        summary = summary_data["summary"]

        # 10. Priority Engine
        priority_result = self.priority_engine.determine_priority(
            text=clean_text,
            category=category,
            sentiment=sentiment,
            compound_score=compound_score,
        )
        priority = priority_result["priority"]
        priority_reasons = priority_result["reasons"]

        # 11. Resolution & Suggested Customer Response
        resolution_result = self.resolution_engine.generate_recommendation(
            category=category,
            priority=priority,
            entities=ner_data["all_entities"],
            sentiment=sentiment,
        )

        # 12. Key Semantic Preview (WordNet on prominent nouns)
        prominent_words = [t["word"] for t in pos_results if t["tag"].startswith("NN")][:4]
        wordnet_preview = self.wordnet_analyzer.get_related_complaint_words(prominent_words)

        return {
            "complaint_id": complaint_id,
            "complaint_text": clean_text,
            "category": category,
            "classification": classification_result,
            "sentiment": sentiment,
            "sentiment_score": compound_score,
            "sentiment_details": sentiment_result,
            "priority": priority,
            "priority_score": priority_result["priority_score"],
            "priority_reasons": priority_reasons,
            "summary": summary,
            "summary_details": summary_data,
            "department": resolution_result["department"],
            "recommended_action": resolution_result["recommended_action"],
            "suggested_response": resolution_result["suggested_response"],
            "agent_notes": resolution_result["agent_notes"],
            "entities": ner_data["all_entities"],
            "domain_entities": ner_data["domain_entities"],
            "standard_entities": ner_data["standard_entities"],
            "preprocessing": prep,
            "stemming": stemmed_tokens,
            "bow": bow_data,
            "tfidf": tfidf_data,
            "pos": pos_results,
            "pos_summary": pos_summary,
            "chunking": chunk_data,
            "wordnet_preview": wordnet_preview,
        }
