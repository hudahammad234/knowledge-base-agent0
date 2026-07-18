"""
main.py
Entry point / orchestrator for the AI Knowledge Assistant pipeline:

  index    -> scan a folder, load new/changed documents, chunk, summarize,
              embed and store them (incremental, no full rebuild).
  ask      -> ask a single question against the indexed knowledge base.
  chat     -> interactive multi-turn session with conversation memory.
  evaluate -> run the evaluation question set and produce a report.

Member responsible: Member 4 - Evaluation / Final Integration
(built on top of modules owned by Members 1-3)
"""

import argparse
import os
import sys

from knowledge_base.loader import load_new_or_changed_documents, scan_folder
from knowledge_base.chunker import chunk_document
from knowledge_base.metadata import extract_metadata
from knowledge_base.summarizer import summarize_document
from retriever import VectorStore, build_context
from query_rewriter import rewrite_query
from prompt_builder import build_prompt
from generator import generate_answer
from validator import validate_answer
from memory import ConversationMemory
from export import export_markdown, export_txt

DEFAULT_KB_FOLDER = "data/sample_docs"
DEFAULT_STORE_DIR = "data/vector_store"
MAX_REGENERATION_ATTEMPTS = 2


def _extract_cited_sources(answer: str):
    import re
    m = re.search(r"sources?:\s*(.*)", answer, re.IGNORECASE)
    if not m:
        return []
    docs = []
    for d in m.group(1).split(","):
        d = d.strip().split(" (p.")[0].strip()
        if d and d.lower() != "none":
            docs.append(d)
    return sorted(set(docs))


class AIKnowledgeAssistant:
    def __init__(self, kb_folder: str = DEFAULT_KB_FOLDER, store_dir: str = DEFAULT_STORE_DIR):
        self.kb_folder = kb_folder
        self.store = VectorStore(persist_dir=store_dir)
        self.store.load()
        self.memory = ConversationMemory()
        self.metadata_records = {}

    # ------------------------------------------------------------------
    def index(self, rebuild: bool = False):
        """Requirement #1/#2: automatic indexing + incremental re-indexing."""
        state = {} if rebuild else self.store.get_index_state()
        new_docs = load_new_or_changed_documents(self.kb_folder, state)

        if not new_docs:
            print("[index] No new or changed documents found. Index is up to date.")
            return

        all_new_chunks = []
        for doc in new_docs:
            full_text = "\n".join(p.text for p in doc.pages)
            summary = summarize_document(full_text)  # BONUS: summarization during indexing
            meta = extract_metadata(doc, summary=summary)
            self.metadata_records[doc.document_name] = meta.to_dict()

            chunks = chunk_document(doc)
            all_new_chunks.extend(chunks)
            print(f"[index] Loaded '{doc.document_name}' -> {len(chunks)} chunks. "
                  f"Summary: {summary[:100]}...")

        self.store.add_chunks(all_new_chunks, rebuild=rebuild)

        new_state = self.store.get_index_state()
        for path in scan_folder(self.kb_folder):
            import hashlib
            with open(path, "rb") as f:
                new_state[path] = hashlib.sha256(f.read()).hexdigest()
        self.store.save_index_state(new_state)

        print(f"[index] Indexed {len(new_docs)} document(s), "
              f"{len(all_new_chunks)} new chunk(s). Total chunks in store: "
              f"{len(self.store.chunk_records)}.")

    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = 5, use_memory: bool = True):
        # BONUS agent: decide whether this needs retrieval or can be answered
        # purely from conversation memory.
        if use_memory and not self.memory.needs_retrieval(question):
            answer = self.memory.answer_from_memory(question)
            self.memory.add_turn(question, question, answer, [], 100.0)
            return {
                "question": question,
                "rewritten_query": question,
                "answer": answer,
                "sources": [],
                "confidence": 100.0,
                "validation_passed": True,
                "from_memory": True,
            }

        history_context = self.memory.get_history_text() if use_memory else ""
        rewritten = rewrite_query(question, current_topic=self.memory.current_topic,
                                   history_context=history_context)

        attempt = 0
        result = None
        while attempt <= MAX_REGENERATION_ATTEMPTS:
            chunks = self.store.hybrid_search(rewritten, top_k=top_k)
            context = build_context(chunks)
            prompt = build_prompt(rewritten, context, conversation_context=history_context)
            answer = generate_answer(rewritten, prompt, chunks)
            validation = validate_answer(rewritten, context, answer, chunks)

            if validation.passed or attempt == MAX_REGENERATION_ATTEMPTS:
                cited_sources = _extract_cited_sources(answer)
                if not cited_sources:
                    cited_sources = sorted({c.document_name for c in chunks})
                result = {
                    "question": question,
                    "rewritten_query": rewritten,
                    "answer": answer,
                    "sources": cited_sources,
                    "confidence": validation.confidence,
                    "validation_passed": validation.passed,
                    "validation_notes": validation.notes,
                    "retrieved_chunks": [c.chunk_id for c in chunks],
                    "from_memory": False,
                }
                break
            attempt += 1

        if not result["validation_passed"]:
            result["answer"] += ("\n\n_Note: this answer did not fully pass automated "
                                  "validation after retries; please treat with caution._")

        if use_memory:
            self.memory.add_turn(question, rewritten, result["answer"],
                                  result["sources"], result["confidence"])
        return result

    # ------------------------------------------------------------------
    def export_conversation(self, fmt: str = "markdown", out_dir: str = "exports"):
        if fmt == "markdown":
            path = os.path.join(out_dir, "conversation_export.md")
            return export_markdown(self.memory, path)
        else:
            path = os.path.join(out_dir, "conversation_export.txt")
            return export_txt(self.memory, path)


def _print_result(result):
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    if result["rewritten_query"] != result["question"]:
        print(f"(rewritten as: {result['rewritten_query']})")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)
    print(f"Confidence: {result['confidence']}% | Validation passed: {result['validation_passed']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Knowledge Assistant")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Index documents in the knowledge base folder")
    p_index.add_argument("--folder", default=DEFAULT_KB_FOLDER)
    p_index.add_argument("--rebuild", action="store_true")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=5)

    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat session")

    p_eval = sub.add_parser("evaluate", help="Run the evaluation question set")
    p_eval.add_argument("--questions", default="evaluation/eval_questions.json")
    p_eval.add_argument("--out", default="evaluation/eval_report.md")

    args = parser.parse_args()
    assistant = AIKnowledgeAssistant()

    if args.command == "index":
        assistant.kb_folder = args.folder
        assistant.index(rebuild=args.rebuild)

    elif args.command == "ask":
        result = assistant.ask(args.question, top_k=args.top_k)
        _print_result(result)

    elif args.command == "chat":
        print("AI Knowledge Assistant - interactive chat. Type 'exit' to quit, "
              "'export md' or 'export txt' to export the conversation.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            if question.lower().startswith("export"):
                fmt = "markdown" if "md" in question.lower() else "txt"
                path = assistant.export_conversation(fmt=fmt)
                print(f"Exported to {path}")
                continue
            result = assistant.ask(question)
            _print_result(result)

    elif args.command == "evaluate":
        from evaluation import run_evaluation
        run_evaluation(assistant, args.questions, args.out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
