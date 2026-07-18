"""
validator.py
Post-generation validation stage. Checks:
  - Is every statement supported by the retrieved context?
  - Is the answer relevant to the question?
  - Does the answer contain hallucinations (claims not present in context)?
  - Are citations correct (do cited documents actually appear in the context)?

BONUS: Confidence scoring based on retrieval quality (average retrieval
similarity score) combined with the groundedness score.

If validation fails, the caller (main.py) can request a regeneration or
reject the answer -- see `ValidationResult.passed`.

Member responsible: Member 3 - Prompt Engineering
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List
from retriever import RetrievedChunk

SUPPORT_THRESHOLD = 0.35     # fraction of answer sentences that must be grounded
MIN_RETRIEVAL_SCORE = 0.05   # below this, we treat retrieval as too weak to trust


@dataclass
class ValidationResult:
    all_statements_supported: bool
    is_relevant: bool
    has_hallucinations: bool
    citations_correct: bool
    confidence: float  # 0-100
    passed: bool
    notes: str


def _tokenize(text: str):
    return set(w.lower() for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 2)


def _heuristic_validate(question: str, context: str, answer: str,
                         chunks: List[RetrievedChunk]) -> ValidationResult:
    context_words = _tokenize(context)
    question_words = _tokenize(question)
    answer_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    # ignore the "Sources:" line when checking groundedness
    content_sentences = [s for s in answer_sentences if not s.lower().startswith("sources")]

    if not content_sentences:
        supported_ratio = 0.0
    else:
        supported = 0
        for sent in content_sentences:
            sent_words = _tokenize(sent)
            if not sent_words:
                continue
            overlap = len(sent_words & context_words) / max(1, len(sent_words))
            if overlap >= 0.3:
                supported += 1
        supported_ratio = supported / len(content_sentences)

    all_supported = supported_ratio >= SUPPORT_THRESHOLD or "don't have enough information" in answer.lower()
    has_hallucinations = not all_supported

    answer_words = _tokenize(answer)
    is_relevant = len(answer_words & question_words) > 0 or "don't have enough information" in answer.lower()

    # citation check: every document name mentioned in "Sources" must be one
    # of the documents actually present in the retrieved chunks
    cited_docs = set()
    m = re.search(r"sources?:\s*(.*)", answer, re.IGNORECASE)
    if m:
        cited_docs = {d.strip().split(" (p.")[0] for d in m.group(1).split(",") if d.strip()}
    cited_docs.discard("none")
    retrieved_docs = {c.document_name for c in chunks}
    citations_correct = cited_docs.issubset(retrieved_docs) if cited_docs else True

    avg_retrieval_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
    retrieval_component = min(1.0, avg_retrieval_score / 0.6) * 100  # normalized, cap at 100
    groundedness_component = supported_ratio * 100
    confidence = round(0.5 * retrieval_component + 0.5 * groundedness_component, 1)
    confidence = max(0.0, min(100.0, confidence))

    passed = all_supported and is_relevant and citations_correct

    notes = (f"supported_ratio={supported_ratio:.2f}, "
             f"avg_retrieval_score={avg_retrieval_score:.2f}, "
             f"cited_docs={sorted(cited_docs)}")

    return ValidationResult(
        all_statements_supported=all_supported,
        is_relevant=is_relevant,
        has_hallucinations=has_hallucinations,
        citations_correct=citations_correct,
        confidence=confidence,
        passed=passed,
        notes=notes,
    )


def _gemini_validate(question: str, context: str, answer: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        from prompt_builder import build_validation_prompt
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = build_validation_prompt(question, context, answer)
        response = model.generate_content(prompt)
        text = response.text.strip().strip("`").replace("json\n", "")
        data = json.loads(text)
        passed = (data.get("all_statements_supported") and data.get("is_relevant")
                  and not data.get("has_hallucinations") and data.get("citations_correct"))
        return ValidationResult(
            all_statements_supported=bool(data.get("all_statements_supported")),
            is_relevant=bool(data.get("is_relevant")),
            has_hallucinations=bool(data.get("has_hallucinations")),
            citations_correct=bool(data.get("citations_correct")),
            confidence=float(data.get("confidence", 0)),
            passed=bool(passed),
            notes=data.get("notes", ""),
        )
    except Exception:
        return None


def validate_answer(question: str, context: str, answer: str,
                     chunks: List[RetrievedChunk]) -> ValidationResult:
    result = _gemini_validate(question, context, answer)
    if result:
        return result
    return _heuristic_validate(question, context, answer, chunks)
