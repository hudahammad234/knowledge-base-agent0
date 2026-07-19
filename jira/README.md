# Jira Project Setup

This folder documents the Jira project for the **AI Knowledge Assistant** team
assignment and provides a ready-to-import backlog.

## How to create the project

1. In Jira, create a new **Team-managed Software project** named
   `AI Knowledge Assistant`.
2. Set the workflow to: `Backlog -> To Do -> In Progress -> Code Review -> Testing -> Done`
   (Project settings -> Workflow -> add the "Code Review" and "Testing"
   statuses between "In Progress" and "Done").
3. Import `jira_backlog.csv` via **Project settings -> Import** (Jira's CSV
   importer). Map columns as:
   - `Summary` -> Summary
   - `Description` -> Description
   - `Assignee` -> Assignee
   - `Priority` -> Priority
   - `Due Date` -> Due Date
   - `Story Points` -> Story point estimate
   - `Status` -> Status
   - `Component` -> Component

   (`jira_backlog.csv` here has every task marked `Done` because it reflects
   the completed reference implementation delivered in this repository; when
   your team runs the sprint for real, import the same file with `Status`
   reset to `Backlog` and move cards through the workflow as you work.)
4. Create 4 team members (or reuse existing Jira users) and assign:
   - **Member 1** - Knowledge Base (loader, metadata, chunking, embeddings)
   - **Member 2** - Retrieval (vector database, search, context builder, query rewriting)
   - **Member 3** - Prompt Engineering (prompt design, Gemini integration, validation, memory)
   - **Member 4** - Evaluation (testing, documentation, export feature, GitHub, Jira, final integration)

## Workflow

```
Backlog -> To Do -> In Progress -> Code Review -> Testing -> Done
```

Each task in the backlog includes: **Description, Assignee, Priority, Due
Date, Story Points** as required by the assignment.

## Sprint plan (6 days)

| Day | Focus |
|---|---|
| 1 | Repo/Git Flow setup, document loader |
| 2 | Chunker, metadata, embeddings, incremental indexing, query rewriter, system prompt |
| 3 | Vector store, BM25, hybrid search, context builder, Gemini generation |
| 4 | Validator, confidence scoring, conversation memory, multi-query, eval questions |
| 5 | Retrieval-vs-memory agent, evaluation runner + auto report, export, docs |
| 6 | Final integration, smoke tests, code review, submission |
