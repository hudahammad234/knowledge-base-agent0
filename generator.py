"""
generator.py
Generates the final answer using Gemini, given the prompt produced by
prompt_builder.py.

Production mode: requires GEMINI_API_KEY and network access -> calls
`gemini-1.5-flash` (swap the model name for `gemini-1.5-pro` for higher
quality at higher cost/latency).

Offline / demo mode (automatic fallback, used in this sandbox which has no
network access): a deterministic extractive generator that composes an
answer strictly from the retrieved chunks, so the rest of the pipeline
(validation, citations, confidence scoring, evaluation) can still be
demonstrated end-to-end without an API key.

Member responsible: Member 3 - Prompt Engineering
"""

import os
from typing import List
from retriever import RetrievedChunk


def _gemini_generate(prompt: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None


def _offline_generate(question: str, chunks: List[RetrievedChunk]) -> str:
    """Deterministic, template-based fallback generator used when Gemini is
    unavailable. It never invents information: it only surfaces sentences
    from the retrieved chunks that share vocabulary with the question."""
    if not chunks:
        return ("I don't have enough information in the knowledge base to answer this.\n\n"
                "Sources: none")

    import re
    GENERIC_WORDS = {
        "company", "companys", "policy", "policies", "employee", "employees",
        "does", "what", "which", "the", "for", "are", "how", "have", "with",
        "mention", "available", "knowledge", "base", "information", "about",
    }
    def _normalize(w):
        w = w.lower()
        return w[:-2] if w.endswith("'s") else w

    all_q_words = set(_normalize(w) for w in re.findall(r"[a-zA-Z']+", question) if len(w) > 2)
    q_words = all_q_words - GENERIC_WORDS
    if not q_words:
        q_words = all_q_words  # question was made only of generic words; fall back

    # Only pull sentences from the top-scoring chunks (retrieval already
    # ranked them), so a loosely-related low-ranked chunk can't crowd out
    # the right answer just because it shares a common word like "policy".
    top_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:2]

    scored_sentences = []
    for c in top_chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c.text):
            sent = sent.strip()
            if len(sent.split()) < 3:
                continue
            sent_words = set(_normalize(w) for w in re.findall(r"[a-zA-Z']+", sent))
            if not sent_words:
                continue
            overlap = len(q_words & sent_words) / max(1, len(q_words)) if q_words else 0
            if overlap > 0:
                # sentence-level overlap dominates; retrieval score only
                # breaks ties between similarly-relevant sentences
                combined_score = overlap * 3 + c.score * 0.3
                scored_sentences.append((combined_score, sent, c))

    if not scored_sentences:
        # nothing overlaps the question -> be honest about it
        return ("I don't have enough information in the knowledge base to answer this "
                "specific question.\n\nSources: none")

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    body_lines = []
    used_sources = []
    for _, sent, c in scored_sentences[:5]:
        body_lines.append(f"- {sent}")
        page_str = f" (p.{c.page_number})" if c.page_number else ""
        src = f"{c.document_name}{page_str}"
        if src not in used_sources:
            used_sources.append(src)

    answer = "Based on the available documents:\n" + "\n".join(body_lines)
    answer += "\n\nSources: " + ", ".join(used_sources)
    return answer


def generate_answer(question: str, prompt: str, chunks: List[RetrievedChunk]) -> str:
    gemini_answer = _gemini_generate(prompt)
    if gemini_answer:
        return gemini_answer
    return _offline_generate(question, chunks)
