"""
prompt_builder.py
Builds the final prompt sent to Gemini, combining:
  - a strict system instruction (answer only from context, never invent info,
    say clearly when information is unavailable, produce structured output)
  - the retrieved / deduplicated context
  - recent conversation memory (for follow-up questions)
  - the (rewritten) user question

Member responsible: Member 3 - Prompt Engineering
"""

SYSTEM_INSTRUCTION = """You are an AI Knowledge Assistant. Answer the user's question using ONLY the
information contained in the CONTEXT below.

Rules:
1. Do not use outside knowledge. Do not invent or assume any fact not present in the context.
2. If the context does not contain enough information to answer, clearly say:
   "I don't have enough information in the knowledge base to answer this."
3. Every factual statement in your answer must be traceable to a specific source in the context.
4. Write a professional, well-structured answer (short paragraphs and/or bullet points).
5. At the end of the answer, include a "Sources" line listing the document name(s) and page
   number(s) (if available) you used.
"""


def build_prompt(question: str, context: str, conversation_context: str = "") -> str:
    memory_block = f"\nRECENT CONVERSATION (for reference resolution only):\n{conversation_context}\n" \
        if conversation_context else ""

    prompt = f"""{SYSTEM_INSTRUCTION}
{memory_block}
CONTEXT:
{context if context.strip() else "(No relevant context was retrieved.)"}

QUESTION:
{question}

ANSWER:"""
    return prompt


def build_validation_prompt(question: str, context: str, answer: str) -> str:
    """Prompt used by validator.py when an LLM-based validation pass is
    available (Gemini). Asks the model to check groundedness/hallucination."""
    return f"""You are a strict fact-checking validator for a RAG system.

CONTEXT:
{context}

QUESTION:
{question}

GENERATED ANSWER:
{answer}

Check the answer against the context ONLY and respond in strict JSON with this schema:
{{
  "all_statements_supported": true/false,
  "is_relevant": true/false,
  "has_hallucinations": true/false,
  "citations_correct": true/false,
  "confidence": 0-100,
  "notes": "short explanation"
}}
Respond with JSON only, no extra text."""
