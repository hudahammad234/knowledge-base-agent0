# Architecture

See `architecture_diagram.png` for the rendered diagram. Source (Mermaid,
renders natively on GitHub) is below.

```mermaid
flowchart TD
    A["Documents\n(PDF / DOCX / TXT / MD / CSV)"] --> B["loader.py\nhash-based change detection"]
    B --> C["chunker.py\nparagraph-aware chunks + overlap"]
    C --> D["metadata.py + summarizer.py\n(BONUS: doc summarization)"]
    D --> E["embeddings.py\nGemini embeddings or offline TF-IDF"]
    E --> F["retriever.py\nVectorStore (persisted, incremental)"]

    U["User Question"] --> Q["query_rewriter.py\nrewrite + multi-query (BONUS)"]
    Q --> H["retriever.py\nHybrid Search: semantic + BM25 (BONUS)\nTop-K + de-dup + context builder"]
    F --> H
    M["memory.py\nconversation memory + mini agent (BONUS)"] --> Q
    H --> M

    H --> P["prompt_builder.py\nsystem instructions + context + memory"]
    P --> G["generator.py\nGemini answer generation"]
    G --> V["validator.py\ngroundedness / hallucination / citation checks\n+ confidence scoring (BONUS)"]
    V -- fail --> R["Regenerate (up to N attempts) or Reject"]
    R --> H
    V -- pass --> O["Final Answer + Citations + Confidence"]
    O --> X["export.py\nMarkdown / TXT export"]
```

## Component responsibilities

| Layer | Module(s) | Responsibility |
|---|---|---|
| Ingestion | `knowledge_base/loader.py` | Load PDF/DOCX/TXT/MD/CSV, detect new/changed files via content hash |
| Ingestion | `knowledge_base/chunker.py` | Paragraph-aware chunking with overlap, chunk numbering |
| Ingestion | `knowledge_base/metadata.py`, `knowledge_base/summarizer.py` | Document metadata + BONUS summarization during indexing |
| Indexing | `embeddings.py` | Gemini embeddings (production) or offline TF-IDF (fallback) |
| Indexing | `retriever.py` (`VectorStore`) | Persistent vector store, incremental add, BM25 index |
| Query time | `query_rewriter.py` | Query rewriting, coreference resolution, BONUS multi-query |
| Query time | `retriever.py` (`hybrid_search`) | BONUS hybrid semantic+keyword search, Top-K, dedupe, context builder |
| Query time | `memory.py` | Conversation memory, BONUS retrieval-vs-memory agent |
| Generation | `prompt_builder.py` | System instruction + context + memory assembly |
| Generation | `generator.py` | Gemini answer generation (offline extractive fallback) |
| Validation | `validator.py` | Groundedness / hallucination / citation checks, BONUS confidence scoring |
| Output | `export.py` | Conversation export to Markdown / TXT |
| Evaluation | `evaluation.py` | Runs eval question set, BONUS automatic report generation |
