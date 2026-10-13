"""Model Training Script for ResolveAI.
Trains:
1. TF-IDF + Logistic Regression Classifier (with real metrics evaluation)
2. Local domain Word2Vec continuous embedding model
Saves artifacts into the models/ directory.
Run:
    python scripts/train_model.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so scripts can be executed directly from anywhere
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DATASET_PATH, CLASSIFIER_PATH, VECTORIZER_PATH, METRICS_PATH, WORD2VEC_PATH
from src.classification.classifier import ComplaintClassifier
from src.word2vec.embeddings import ComplaintWord2Vec


def main():
    print("=" * 65)
    print("ResolveAI: Training Machine Learning & Word2Vec Models")
    print("=" * 65)

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}. Please provide a complaints dataset before training.")
        return

    print("\n[1/2] Training TF-IDF Classifier on sample complaints dataset...")
    classifier = ComplaintClassifier(
        classifier_path=CLASSIFIER_PATH,
        vectorizer_path=VECTORIZER_PATH,
        metrics_path=METRICS_PATH,
    )
    metrics = classifier.train(dataset_path=DATASET_PATH)

    print(f"  ✓ Classifier trained successfully!")
    print(f"  ✓ Total samples: {metrics['total_samples']}")
    print(f"  ✓ Train split:   {metrics['train_samples']} samples")
    print(f"  ✓ Test split:    {metrics['test_samples']} samples")
    print(f"  ✓ Test Accuracy: {metrics['accuracy_percentage']}%")
    print(f"  ✓ Macro F1:      {metrics['macro_f1']}")
    print(f"  ✓ Saved to:      {CLASSIFIER_PATH} and {VECTORIZER_PATH}")

    print("\n[2/2] Training local Word2Vec continuous embedding model...")
    w2v = ComplaintWord2Vec(model_path=WORD2VEC_PATH)
    w2v.train_and_save(csv_path=DATASET_PATH)
    print(f"  ✓ Word2Vec model trained successfully!")
    print(f"  ✓ Vocabulary size: {len(w2v.model.wv.index_to_key)} words")
    print(f"  ✓ Embedding vector dimension: {w2v.model.vector_size}")
    print(f"  ✓ Saved to: {WORD2VEC_PATH}")

    print("\n" + "=" * 65)
    print("Training Complete! Models are ready for ResolveAI.")
    print("=" * 65)


if __name__ == "__main__":
    main()
