# AI Knowledge Assistant

**AI Track — Team Assignment 02**
Team: 2 teams x 4 members. This repository is the reference implementation for
**Team A** (Team B would fork/adapt the same skeleton independently).

An AI-engineered Retrieval-Augmented Generation (RAG) system that indexes a
document collection (PDF, DOCX, TXT, Markdown, CSV), retrieves the most
relevant chunks using **hybrid semantic + keyword search**, generates
grounded answers with Gemini, validates every answer against the retrieved
context, and reports a confidence score with full source citations.

> This build runs in two modes automatically:
> - **Production mode** — set `GEMINI_API_KEY` (see `.env.example`) to use real
>   Gemini generation, embeddings, query rewriting and validation.
> - **Offline / demo mode** — with no key (e.g. no internet / no API key,
>   as in this sandboxed build), the pipeline automatically falls back to a
>   TF-IDF embedder, a BM25 keyword index, and a deterministic extractive
>   generator, so the **entire pipeline still runs end-to-end** and can be
>   graded/tested without any external dependency.

## Team Responsibilities

| Member | Focus Area | Responsibilities |
|---|---|---|
| Member 1 | Knowledge Base | `loader.py`, `metadata.py`, `chunker.py`, `summarizer.py`, `embeddings.py` |
| Member 2 | Retrieval | `retriever.py` (vector store, BM25, hybrid search, context builder), `query_rewriter.py` |
| Member 3 | Prompt Engineering | `prompt_builder.py`, `generator.py`, `validator.py`, `memory.py` |
| Member 4 | Evaluation | `evaluation.py`, `export.py`, GitHub, Jira, `main.py` final integration, docs |

## Project Structure

```
project/
├── knowledge_base/
│   ├── loader.py        # PDF/DOCX/TXT/MD/CSV loading + change detection
│   ├── chunker.py        # paragraph-aware chunking with overlap
│   ├── metadata.py        # document metadata extraction
│   └── summarizer.py       # BONUS: document summarization during indexing
├── embeddings.py          # Gemini embeddings or offline TF-IDF fallback
├── retriever.py           # vector store, BM25, BONUS hybrid search, context builder
├── query_rewriter.py      # query rewriting + BONUS multi-query retrieval
├── prompt_builder.py       # system prompt + context assembly
├── generator.py           # Gemini answer generation (+ offline fallback)
├── validator.py           # groundedness/hallucination/citation checks + BONUS confidence scoring
├── memory.py              # conversation memory + BONUS retrieval-vs-memory agent
├── export.py              # conversation export (Markdown/TXT)
├── evaluation.py          # evaluation runner + BONUS automatic report generation
├── main.py                # CLI orchestrator (index / ask / chat / evaluate)
├── data/
│   ├── sample_docs/        # sample knowledge base (pdf, docx, txt, md, csv)
│   └── vector_store/        # persisted index (auto-created)
├── evaluation/
│   ├── eval_questions.json  # 30 evaluation questions
│   └── eval_report.md       # auto-generated evaluation report (+ .json/.csv)
├── exports/                # sample conversation export files
├── architecture/           # architecture diagram (png + mermaid source)
├── jira/                  # Jira backlog CSV + setup instructions
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GEMINI_API_KEY for production mode
```

## Usage

```bash
# 1. Index the sample knowledge base (incremental — safe to re-run)
python3 main.py index

# 2. Ask a single question
python3 main.py ask "What is the company's vacation policy?"

# 3. Interactive multi-turn chat (type 'exit' to quit, 'export md'/'export txt' to export)
python3 main.py chat

# 4. Run the 30-question evaluation suite and auto-generate the report
python3 main.py evaluate
```

To index your own documents, drop PDF/DOCX/TXT/MD/CSV files into
`data/sample_docs/` (or point `--folder` elsewhere) and re-run
`python3 main.py index` — only new or changed files are processed.

## How each requirement is satisfied

1. **Intelligent Knowledge Base** — `loader.py` auto-detects PDF/DOCX/TXT/MD/CSV
   and uses a file-hash index (`data/vector_store/index_state.json`) to only
   load new/changed files; `VectorStore.add_chunks(rebuild=False)` appends
   without rebuilding the whole index.
2. **Advanced Document Processing** — `chunker.py` splits on paragraph
   boundaries into ~220-word overlapping chunks; `metadata.py` stores
   document name, page number, chunk number, size, hash, and (bonus) summary
   with every chunk.
3. **Intelligent Query Rewriting** — `query_rewriter.py`, documented in the
   module docstring: Gemini-based rewriting in production, rule-based
   keyword-expansion + coreference resolution offline.
4. **Semantic Retrieval** — `retriever.hybrid_search()` retrieves Top-K,
   removes duplicates, and `build_context()` caps total context size.
5. **AI Answer Generation** — `generator.py` + `prompt_builder.py` enforce
   context-only, no-invention, "I don't know" fallback, structured answers.
6. **Response Validation** — `validator.py` checks support, relevance,
   hallucination, and citation correctness; `main.py` regenerates up to 2
   times before rejecting/flagging the answer.
7. **Source Citations** — every answer ends with a `Sources:` line
   (document + page), and `evaluation.py`/`export.py` report a confidence %.
8. **Conversation Memory** — `memory.py` tracks turns and resolves follow-up
   references (e.g. "what about maternity leave?").
9. **Conversation Export** — `export.py` writes Markdown and TXT transcripts;
   samples are in `exports/`.
10. **AI Evaluation** — `evaluation/eval_questions.json` has 30 questions;
    `evaluation.py` scores retrieval correctness, groundedness, and an
    overall score, and auto-writes `eval_report.md/.json/.csv`.

## Bonus features implemented

- ✅ **Hybrid Search** — `retriever.hybrid_search()` fuses semantic cosine
  similarity with a from-scratch BM25 keyword index.
- ✅ **Document summarization during indexing** — `knowledge_base/summarizer.py`.
- ✅ **Confidence scoring based on retrieval quality** — `validator.py`
  blends average retrieval score with groundedness ratio.
- ✅ **Automatic evaluation report generation** — `evaluation.py` writes a
  Markdown report with summary stats + per-question table + full Q&A log.
- ✅ **Multi-query retrieval** — `query_rewriter.rewrite_multi()`.
- ✅ **Simple retrieval-vs-memory agent** — `memory.needs_retrieval()` /
  `memory.answer_from_memory()`.

## GitHub workflow

This repo is initialized with Git Flow: `main` (releases) and `develop`
(integration), with each module built on its own `feature/*` branch and
merged via a (locally simulated) Pull Request — see `git log --graph --all`
for the full history and `.github/PULL_REQUEST_TEMPLATE.md` for the PR
checklist used by the team. Push this repo to GitHub and open the branches
as real PRs to complete the review workflow with your teammates.

## Notes on the offline fallback

Because Gemini requires network access and an API key, this reference build
was validated **fully offline**. Every module that would call Gemini in
production (embeddings, query rewriting, generation, validation,
summarization) has an automatic, clearly-documented offline fallback so the
grader can run `index`, `ask`, `chat`, and `evaluate` end-to-end with zero
setup. Simply add `GEMINI_API_KEY` to switch every one of those modules to
the real Gemini-backed implementation with no code changes.
