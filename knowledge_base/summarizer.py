"""
summarizer.py
BONUS FEATURE: Document summarization during indexing.

If a Gemini API key is configured (GEMINI_API_KEY) and the environment has
network access, the document is summarized with Gemini. Otherwise the module
falls back to a lightweight, fully offline extractive summarizer (word
frequency scoring, similar in spirit to TextRank) so the pipeline still works
with no external dependency.

Member responsible: Member 1 - Knowledge Base
"""

import os
import re
from collections import Counter

STOPWORDS = set("""
a an the this that these those is are was were be been being to of in on at
for with by from as it its it's and or but if then than so such not no nor
we you they he she i my our your their his her them us do does did will
would can could should may might must have has had here there when where
which who whom what why how all any some more most other into over under
again further once about above below during before after up down out off
""".split())


def _extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    if not sentences:
        return text[:280].strip()
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    freqs = Counter(w for w in words if w not in STOPWORDS)
    max_freq = max(freqs.values()) if freqs else 1

    scored = []
    for idx, sent in enumerate(sentences):
        sent_words = re.findall(r"[a-zA-Z']+", sent.lower())
        score = sum(freqs.get(w, 0) for w in sent_words) / max_freq
        # slight bias toward earlier sentences (titles / intros carry signal)
        score += max(0, (10 - idx)) * 0.01
        scored.append((score, idx, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_in_order = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_in_order)


def _gemini_summary(text: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Summarize the following document in 2-3 sentences for use as "
            "search-index metadata. Be factual, do not add outside "
            "information.\n\nDOCUMENT:\n" + text[:8000]
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None


def summarize_document(full_text: str, max_sentences: int = 3) -> str:
    """Return a short summary of a document's full text (used as metadata)."""
    if not full_text.strip():
        return ""
    summary = _gemini_summary(full_text)
    if summary:
        return summary
    return _extractive_summary(full_text, max_sentences=max_sentences)
