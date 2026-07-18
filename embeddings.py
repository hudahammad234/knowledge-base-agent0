"""
embeddings.py
Generates vector embeddings for chunks and queries.

Production note
----------------
This module is written with a pluggable backend:
  1. GEMINI  -> uses Google's text-embedding-004 model via google.generativeai
               when GEMINI_API_KEY is set and network access is available.
  2. TFIDF   -> a fully offline TF-IDF vectorizer (implemented from scratch
               with numpy, no external ML dependency required) used as the
               automatic fallback so the assistant works in restricted /
               offline environments and in CI.

Both backends expose the same interface: fit(texts), encode(texts) -> np.ndarray.

Member responsible: Member 1 - Knowledge Base / Member 2 - Retrieval
"""

import os
import re
import math
import pickle
from typing import List
import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z\u0600-\u06FF']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class TfidfEmbedder:
    """A minimal, dependency-free TF-IDF vectorizer used as the offline
    embedding backend. Produces L2-normalized vectors so cosine similarity
    reduces to a dot product."""

    def __init__(self):
        self.vocab = {}
        self.idf = None

    def fit(self, texts: List[str]):
        doc_freq = {}
        for text in texts:
            seen = set(tokenize(text))
            for tok in seen:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + doc_freq[tok])) + 1
        return self

    def _vectorize_one(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec
        tf_counts = {}
        for tok in tokens:
            if tok in self.vocab:
                tf_counts[tok] = tf_counts.get(tok, 0) + 1
        for tok, count in tf_counts.items():
            idx = self.vocab[tok]
            tf = count / len(tokens)
            vec[idx] = tf * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.vocab is None or len(self.vocab) == 0:
            raise RuntimeError("TfidfEmbedder must be fit() before encode()")
        return np.stack([self._vectorize_one(t) for t in texts])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "idf": self.idf}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        return self


def _gemini_embed(texts: List[str]):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        vectors = []
        for t in texts:
            result = genai.embed_content(model="models/text-embedding-004", content=t)
            vectors.append(result["embedding"])
        return np.array(vectors, dtype=np.float32)
    except Exception:
        return None


class Embedder:
    """Facade used by the rest of the pipeline. Tries Gemini embeddings first,
    transparently falls back to the offline TF-IDF embedder."""

    def __init__(self, store_path: str = "data/vector_store/tfidf_embedder.pkl"):
        self.store_path = store_path
        self.tfidf = TfidfEmbedder()
        self.backend = "tfidf"

    def fit(self, texts: List[str]):
        self.tfidf.fit(texts)
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.tfidf.save(self.store_path)
        return self

    def load(self):
        if os.path.exists(self.store_path):
            self.tfidf.load(self.store_path)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        gemini_vecs = _gemini_embed(texts)
        if gemini_vecs is not None:
            self.backend = "gemini"
            return gemini_vecs
        self.backend = "tfidf"
        return self.tfidf.encode(texts)
