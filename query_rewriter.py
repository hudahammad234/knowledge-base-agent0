"""
query_rewriter.py
Rewrites short / ambiguous user questions into complete, retrieval-friendly
search queries before hitting the vector store, e.g.:

    "Vacation" -> "What are the company's annual leave and vacation policies?"

Approach (documented per assignment requirement #3)
----------------------------------------------------
1. If GEMINI_API_KEY is configured, the raw question (plus the last N turns
   of conversation memory for coreference resolution, e.g. "what about
   maternity leave?" after a question about annual leave) is sent to Gemini
   with an instruction to produce ONE complete, self-contained search query.
2. Offline fallback: a rule-based expander that
     a. Resolves obvious pronouns/references using the previous turn's topic
        (via memory.py's tracked "current_topic").
     b. Expands very short (<=3 word) queries using a small domain synonym
        map, turning single keywords into a fuller natural-language question.
     c. Otherwise returns the question unchanged (it is already a full
        sentence and does not need rewriting).

BONUS: Multi-query retrieval is supported via `rewrite_multi`, which asks for
several alternative phrasings of the same question to broaden recall.

Member responsible: Member 2 - Retrieval
"""

import os
import re
from typing import List, Optional

# small domain vocabulary map used by the offline fallback to flesh out
# single-keyword queries into fuller natural-language questions.
KEYWORD_EXPANSIONS = {
    "vacation": "What are the company's annual leave and vacation policies?",
    "leave": "What are the company's leave policies (annual, sick, and other leave types)?",
    "maternity": "What is the company's maternity leave policy?",
    "paternity": "What is the company's paternity leave policy?",
    "salary": "How is employee salary and compensation structured?",
    "benefits": "What employee benefits does the company offer?",
    "remote": "What is the company's remote / work-from-home policy?",
    "overtime": "What is the company's overtime policy and compensation?",
    "onboarding": "What is the employee onboarding process?",
    "termination": "What is the company's termination and resignation policy?",
    "expenses": "What is the company's expense reimbursement policy?",
    "security": "What are the company's data and information security policies?",
}


def _gemini_rewrite(question: str, history_context: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Rewrite the user's question into ONE complete, self-contained "
            "search query suitable for retrieving relevant document chunks. "
            "Resolve pronouns/references using the conversation history. "
            "Return ONLY the rewritten query, nothing else.\n\n"
            f"Conversation history:\n{history_context}\n\n"
            f"User question: {question}"
        )
        response = model.generate_content(prompt)
        return response.text.strip().strip('"')
    except Exception:
        return None


def _rule_based_rewrite(question: str, current_topic: Optional[str]) -> str:
    q = question.strip()
    words = q.split()

    # Very short / single-keyword query -> expand using domain map
    if len(words) <= 3:
        key = q.lower().strip("? .").split()[-1] if words else ""
        if key in KEYWORD_EXPANSIONS:
            return KEYWORD_EXPANSIONS[key]

    # Follow-up reference resolution, e.g. "what about maternity leave?"
    followup_markers = ("what about", "and", "also", "what's", "how about")
    if current_topic and q.lower().startswith(followup_markers):
        if not q.endswith("?"):
            q += "?"
        return f"{q} (in the context of: {current_topic})"

    # already a full question/sentence
    if q.endswith("?") or len(words) >= 5:
        return q

    return f"What information is available about {q}?"


def rewrite_query(question: str, current_topic: Optional[str] = None,
                   history_context: str = "") -> str:
    """Main entry point used by main.py before retrieval."""
    rewritten = _gemini_rewrite(question, history_context)
    if rewritten:
        return rewritten
    return _rule_based_rewrite(question, current_topic)


def rewrite_multi(question: str, n: int = 3, current_topic: Optional[str] = None) -> List[str]:
    """BONUS: multi-query retrieval - generate several alternative phrasings
    of the same question to broaden recall. Offline fallback simply combines
    the rewritten query with light lexical variants."""
    base = rewrite_query(question, current_topic=current_topic)
    variants = [base]

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Generate {n - 1} alternative search-query phrasings of this "
                f"question, one per line, no numbering:\n{base}"
            )
            response = model.generate_content(prompt)
            extra = [l.strip("- ").strip() for l in response.text.split("\n") if l.strip()]
            variants.extend(extra[: n - 1])
            return variants
        except Exception:
            pass

    # offline fallback variants: keyword-only + question form
    words = re.findall(r"[a-zA-Z']+", base)
    if words:
        variants.append(" ".join(words[:6]))
    variants.append(question)
    # de-duplicate, preserve order
    seen = set()
    unique = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique.append(v)
    return unique[:n]
