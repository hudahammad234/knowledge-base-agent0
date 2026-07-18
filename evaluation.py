"""
evaluation.py
Requirement #10 (AI Evaluation) + BONUS (Automatic evaluation report generation).

Runs a set of >=25 evaluation questions against the live pipeline and records,
for each one:
  - expected_answer   (ground truth, authored by the team from the sample KB)
  - actual_answer     (what the pipeline produced)
  - retrieval_correct (did the retrieved chunks come from the expected
                       source document(s)?)
  - answer_grounded   (did validator.py consider the answer grounded?)
  - overall_score     (0-100, combining retrieval correctness, groundedness,
                       and confidence)

Produces both a machine-readable JSON/CSV and a human-readable Markdown
report (auto-generated -- the bonus feature).

Member responsible: Member 4 - Evaluation
"""

import json
import os
import csv
from datetime import datetime


def _load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_case(case, result, validation_notes: str) -> dict:
    expected_docs = set(case.get("expected_sources", []))
    actual_docs = set(result.get("sources", []))
    retrieval_correct = bool(expected_docs & actual_docs) if expected_docs else True

    answer_grounded = result.get("validation_passed", False)

    confidence = result.get("confidence", 0)
    score = 0.0
    score += 40 if retrieval_correct else 0
    score += 40 if answer_grounded else 0
    score += min(20, confidence / 5)  # confidence contributes up to 20 points
    overall_score = round(score, 1)

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "actual_answer": result["answer"],
        "expected_sources": sorted(expected_docs),
        "actual_sources": sorted(actual_docs),
        "retrieval_correct": retrieval_correct,
        "answer_grounded": answer_grounded,
        "confidence": confidence,
        "overall_score": overall_score,
    }


def run_evaluation(assistant, questions_path: str, report_path: str):
    cases = _load_questions(questions_path)
    results = []

    for case in cases:
        # each evaluation question is asked as a fresh, independent turn
        assistant.memory.turns = []
        assistant.memory.current_topic = None
        result = assistant.ask(case["question"], use_memory=False)
        scored = _score_case(case, result, result.get("validation_notes", ""))
        results.append(scored)
        print(f"[eval] {case['id']}: score={scored['overall_score']} "
              f"retrieval_correct={scored['retrieval_correct']} "
              f"grounded={scored['answer_grounded']}")

    _write_report(results, report_path)
    _write_json_csv(results, report_path)
    return results


def _write_json_csv(results, report_path: str):
    base, _ = os.path.splitext(report_path)
    json_path = base + ".json"
    csv_path = base + ".csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def _write_report(results, report_path: str):
    n = len(results)
    avg_score = round(sum(r["overall_score"] for r in results) / n, 1) if n else 0
    retrieval_acc = round(100 * sum(r["retrieval_correct"] for r in results) / n, 1) if n else 0
    grounded_rate = round(100 * sum(r["answer_grounded"] for r in results) / n, 1) if n else 0
    avg_conf = round(sum(r["confidence"] for r in results) / n, 1) if n else 0

    lines = [
        "# AI Knowledge Assistant - Evaluation Report",
        f"_Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        f"- Total questions evaluated: **{n}**",
        f"- Average overall score: **{avg_score} / 100**",
        f"- Retrieval accuracy: **{retrieval_acc}%**",
        f"- Grounded-answer rate: **{grounded_rate}%**",
        f"- Average confidence: **{avg_conf}%**",
        "",
        "## Per-Question Results",
        "",
        "| # | Question | Retrieval OK | Grounded | Confidence | Score |",
        "|---|----------|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['question']} | "
            f"{'✅' if r['retrieval_correct'] else '❌'} | "
            f"{'✅' if r['answer_grounded'] else '❌'} | "
            f"{r['confidence']}% | {r['overall_score']} |"
        )

    lines.append("")
    lines.append("## Detailed Q&A Log")
    for r in results:
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append(f"**Expected answer:** {r['expected_answer']}")
        lines.append("")
        lines.append(f"**Actual answer:**\n\n{r['actual_answer']}")
        lines.append("")
        lines.append(f"**Expected sources:** {', '.join(r['expected_sources']) or 'n/a'}  ")
        lines.append(f"**Actual sources:** {', '.join(r['actual_sources']) or 'none'}  ")
        lines.append(f"**Score:** {r['overall_score']} / 100")
        lines.append("\n---\n")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[eval] Report written to {report_path}")
    print(f"[eval] Average score: {avg_score}/100 | Retrieval accuracy: {retrieval_acc}% | "
          f"Grounded rate: {grounded_rate}%")
