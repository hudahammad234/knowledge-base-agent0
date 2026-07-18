"""
retriever.py
Vector database + retrieval layer.

Implements:
  - A lightweight persistent vector store (numpy + JSON, no external service
    required, but structured so it could be swapped for Chroma/FAISS/Pinecone
    in production by re-implementing this class's interface).
  - Semantic retrieval (embedding cosine similarity).
  - BONUS: Hybrid search = semantic similarity + BM25 keyword search, fused
    with a weighted-sum re-ranker.
  - Top-K selection, de-duplication, and intelligent context building that
    avoids sending unnecessary text to Gemini.

Member responsible: Member 2 - Retrieval
"""

import os
import json
import math
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

from embeddings import Embedder, tokenize


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_name: str
    page_number: Optional[int]
    chunk_number: int
    text: str
    score: float
    semantic_score: float
    keyword_score: float


class BM25:
    """Minimal, dependency-free BM25 keyword search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avgdl = 0.0
        self.N = 0

    def fit(self, texts: List[str]):
        self.doc_tokens = [tokenize(t) for t in texts]
        self.N = len(self.doc_tokens)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / max(1, self.N)
        self.doc_freqs = {}
        for tokens in self.doc_tokens:
            for tok in set(tokens):
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1
        return self

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.doc_tokens):
            if not doc:
                continue
            dl = len(doc)
            freqs = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            s = 0.0
            for term in q_tokens:
                if term not in freqs:
                    continue
                idf = self._idf(term)
                f = freqs[term]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores[i] = s
        max_s = scores.max() if scores.size else 0.0
        if max_s > 0:
            scores = scores / max_s  # normalize to [0,1] for fair fusion with cosine sim
        return scores


class VectorStore:
    """Persistent store for chunk metadata + embeddings + BM25 index."""

    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.chunks_path = os.path.join(persist_dir, "chunks.json")
        self.vectors_path = os.path.join(persist_dir, "vectors.npy")
        self.bm25_path = os.path.join(persist_dir, "bm25.pkl")
        self.index_state_path = os.path.join(persist_dir, "index_state.json")

        self.chunk_records: List[Dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.bm25: Optional[BM25] = None
        self.embedder = Embedder(store_path=os.path.join(persist_dir, "tfidf_embedder.pkl"))

    # -- persistence -----------------------------------------------------
    def load(self):
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.chunk_records = json.load(f)
        if os.path.exists(self.vectors_path):
            self.vectors = np.load(self.vectors_path)
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        self.embedder.load()
        return self

    def _save(self):
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_records, f, ensure_ascii=False, indent=2)
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)
        if self.bm25 is not None:
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)

    def get_index_state(self) -> Dict[str, str]:
        if os.path.exists(self.index_state_path):
            with open(self.index_state_path, "r") as f:
                return json.load(f)
        return {}

    def save_index_state(self, state: Dict[str, str]):
        with open(self.index_state_path, "w") as f:
            json.dump(state, f, indent=2)

    # -- indexing ----------------------------------------------------------
    def add_chunks(self, chunks, rebuild: bool = False):
        """Add new chunks to the store. If rebuild=False (default) this
        APPENDS to the existing index and only re-fits the embedder/BM25 over
        the full corpus (needed because TF-IDF/BM25 statistics are corpus-wide)
        -- but it does NOT require re-reading or re-chunking any previously
        indexed documents, satisfying the 'no full rebuild' requirement."""
        new_records = [asdict(c) for c in chunks]
        if rebuild:
            self.chunk_records = new_records
        else:
            existing_ids = {r["chunk_id"] for r in self.chunk_records}
            self.chunk_records.extend(r for r in new_records if r["chunk_id"] not in existing_ids)

        all_texts = [r["text"] for r in self.chunk_records]
        if not all_texts:
            return

        self.embedder.fit(all_texts)
        self.vectors = self.embedder.encode(all_texts)

        self.bm25 = BM25().fit(all_texts)
        self._save()

    # -- search --------------------------------------------------------
    def semantic_search(self, query: str, top_k: int) -> np.ndarray:
        if self.vectors is None or len(self.chunk_records) == 0:
            return np.array([])
        q_vec = self.embedder.encode([query])[0]
        sims = self.vectors @ q_vec  # vectors are L2-normalized -> dot = cosine
        return sims

    def keyword_search(self, query: str) -> np.ndarray:
        if self.bm25 is None or len(self.chunk_records) == 0:
            return np.array([])
        return self.bm25.score_all(query)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[RetrievedChunk]:
        """BONUS: Hybrid search combining semantic similarity (weight=alpha)
        and BM25 keyword score (weight=1-alpha), then deduplicating and
        returning the Top-K unique chunks."""
        if not self.chunk_records:
            return []

        sem_scores = self.semantic_search(query, top_k)
        kw_scores = self.keyword_search(query)

        combined = alpha * sem_scores + (1 - alpha) * kw_scores
        order = np.argsort(-combined)

        results: List[RetrievedChunk] = []
        seen_texts = set()
        for idx in order:
            record = self.chunk_records[idx]
            dedup_key = record["text"].strip()[:300]
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)
            results.append(RetrievedChunk(
                chunk_id=record["chunk_id"],
                document_name=record["document_name"],
                page_number=record["page_number"],
                chunk_number=record["chunk_number"],
                text=record["text"],
                score=float(combined[idx]),
                semantic_score=float(sem_scores[idx]) if len(sem_scores) else 0.0,
                keyword_score=float(kw_scores[idx]) if len(kw_scores) else 0.0,
            ))
            if len(results) >= top_k:
                break
        return results


def build_context(chunks: List[RetrievedChunk], max_words: int = 900) -> str:
    """Builds an intelligently truncated context block from retrieved chunks,
    avoiding sending unnecessary information to Gemini. Chunks are already
    deduplicated by the vector store; here we additionally cap total length
    and stop including further chunks once the budget is spent."""
    parts = []
    word_budget = max_words
    for c in chunks:
        words = c.text.split()
        if word_budget <= 0:
            break
        take = words[:word_budget]
        word_budget -= len(take)
        page_str = f", page {c.page_number}" if c.page_number else ""
        parts.append(
            f"[Source: {c.document_name}{page_str}, chunk #{c.chunk_number}]\n{' '.join(take)}"
        )
    return "\n\n---\n\n".join(parts)
