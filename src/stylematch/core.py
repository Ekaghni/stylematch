"""Core comparison logic.

The method is the usual stylometry trick: break each text into overlapping
character n-grams (2 to 4 characters long), weight them with TF-IDF, and take
the cosine similarity of the two resulting vectors. Character patterns carry
habits like punctuation, spacing and common word endings, so the score leans
on style more than on topic.

Everything here is plain Python. No numpy, no scikit-learn, no GPU.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

NGRAM_MIN = 2
NGRAM_MAX = 4
MIN_WORDS = 10

HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.50

_WHITESPACE = re.compile(r"\s\s+")


def _ngrams(text: str) -> Counter:
    """Count character n-grams, lowercased, with runs of whitespace collapsed."""
    text = _WHITESPACE.sub(" ", text.lower())
    counts: Counter = Counter()
    length = len(text)
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(length - n + 1):
            counts[text[i : i + n]] += 1
    return counts


def _tfidf(docs: List[Counter]) -> List[Dict[str, float]]:
    """Smoothed TF-IDF with L2 normalisation (same recipe scikit-learn uses)."""
    n_docs = len(docs)
    doc_freq: Counter = Counter()
    for doc in docs:
        doc_freq.update(doc.keys())

    vectors = []
    for doc in docs:
        vec = {}
        for gram, tf in doc.items():
            idf = math.log((1 + n_docs) / (1 + doc_freq[gram])) + 1
            vec[gram] = tf * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm:
            vec = {g: v / norm for g, v in vec.items()}
        vectors.append(vec)
    return vectors


def similarity(text1: str, text2: str) -> float:
    """Return the style similarity of two texts as a number from 0.0 to 1.0."""
    if not text1 or not text1.strip() or not text2 or not text2.strip():
        raise ValueError("Both texts must contain something to compare.")

    a, b = _tfidf([_ngrams(text1), _ngrams(text2)])
    if len(a) > len(b):
        a, b = b, a
    score = sum(weight * b.get(gram, 0.0) for gram, weight in a.items())
    # Floating point can push a perfect match a hair past 1.
    return max(0.0, min(1.0, score))


def interpret(score: float) -> str:
    """Turn a score into a short plain-English reading."""
    if score >= HIGH_THRESHOLD:
        return "Very similar - likely the same author"
    if score >= MEDIUM_THRESHOLD:
        return "Somewhat similar - possibly the same author, or related topics"
    return "Different - likely different authors or topics"


@dataclass
class Comparison:
    """Everything `compare` finds out about a pair of texts."""

    score: float
    verdict: str
    words_1: int
    words_2: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "words_1": self.words_1,
            "words_2": self.words_2,
            "warnings": list(self.warnings),
        }


def compare(text1: str, text2: str) -> Comparison:
    """Compare two texts and return the score, a verdict and some context."""
    score = similarity(text1, text2)
    words_1, words_2 = len(text1.split()), len(text2.split())

    warnings = []
    if words_1 < MIN_WORDS or words_2 < MIN_WORDS:
        warnings.append(
            f"Texts should have at least {MIN_WORDS} words each. "
            "Short samples give unreliable scores."
        )
    return Comparison(score, interpret(score), words_1, words_2, warnings)
