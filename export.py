"""
export.py
Allows users to export the full conversation as Markdown or plain text.

Member responsible: Member 4 - Evaluation / Integration
"""

import os
import time
from memory import ConversationMemory


def export_markdown(memory: ConversationMemory, out_path: str) -> str:
    lines = [f"# Conversation Export", f"_Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n"]
    for i, t in enumerate(memory.turns, start=1):
        lines.append(f"## Turn {i}")
        lines.append(f"**User:** {t.question}\n")
        lines.append(f"**Rewritten query:** {t.rewritten_question}\n")
        lines.append(f"**Assistant:** {t.answer}\n")
        lines.append(f"**Confidence:** {t.confidence}%\n")
        lines.append(f"**Sources:** {', '.join(t.sources) if t.sources else 'none'}\n")
        lines.append("---\n")
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def export_txt(memory: ConversationMemory, out_path: str) -> str:
    lines = [f"Conversation Export - {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 50, ""]
    for i, t in enumerate(memory.turns, start=1):
        lines.append(f"Turn {i}")
        lines.append(f"User: {t.question}")
        lines.append(f"Rewritten query: {t.rewritten_question}")
        lines.append(f"Assistant: {t.answer}")
        lines.append(f"Confidence: {t.confidence}%")
        lines.append(f"Sources: {', '.join(t.sources) if t.sources else 'none'}")
        lines.append("-" * 50)
        lines.append("")
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path
