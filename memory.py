"""
memory.py
Conversation memory for multi-turn sessions. Tracks turns so the assistant
understands references to previous questions (e.g. a follow-up about
"maternity leave" after asking about the annual leave policy), and exposes a
formatted history block for the prompt.

BONUS: a very small "agent" heuristic (`needs_retrieval`) that decides
whether a new question can likely be answered purely from conversation
memory (e.g. "what did I just ask?") or needs a fresh retrieval pass.

Member responsible: Member 3 - Prompt Engineering
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Turn:
    question: str
    rewritten_question: str
    answer: str
    sources: List[str]
    confidence: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.turns: List[Turn] = []
        self.max_turns = max_turns
        self.current_topic: Optional[str] = None

    def add_turn(self, question: str, rewritten_question: str, answer: str,
                 sources: List[str], confidence: float):
        self.turns.append(Turn(question, rewritten_question, answer, sources, confidence))
        self.current_topic = rewritten_question
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history_text(self, last_n: int = 3) -> str:
        recent = self.turns[-last_n:]
        lines = []
        for t in recent:
            lines.append(f"User: {t.question}\nAssistant: {t.answer[:300]}")
        return "\n\n".join(lines)

    def needs_retrieval(self, question: str) -> bool:
        """BONUS: simple agent that decides whether a question can be
        answered purely from conversation memory (meta questions about the
        chat itself) or needs a fresh document retrieval pass."""
        q = question.lower().strip()
        meta_patterns = [
            r"what did i (just )?ask", r"what was my (last|previous) question",
            r"repeat (that|your last answer)", r"summarize (this|our) conversation",
            r"what have we (talked|discussed) about",
        ]
        for pattern in meta_patterns:
            if re.search(pattern, q):
                return False
        return True

    def answer_from_memory(self, question: str) -> str:
        q = question.lower()
        if not self.turns:
            return "We haven't discussed anything yet in this session."
        if re.search(r"what did i (just )?ask|what was my (last|previous) question", q):
            return f'Your last question was: "{self.turns[-1].question}"'
        if re.search(r"repeat", q):
            return self.turns[-1].answer
        if re.search(r"summarize", q):
            topics = ", ".join(t.rewritten_question for t in self.turns)
            return f"So far in this conversation we covered: {topics}"
        return "I can only answer that from conversation memory in a limited way."

    def export_dict(self):
        return [t.__dict__ for t in self.turns]
