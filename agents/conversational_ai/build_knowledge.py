"""Build a development-only baseline retriever for SmartBank synthetic knowledge assets."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, write_json, write_model_card  # noqa: E402
from agents.conversational_ai.safety import safety_response  # noqa: E402


def build(knowledge_path: str, evaluation_path: str, output_dir: str) -> dict[str, float | int | bool]:
    articles = [json.loads(line) for line in Path(knowledge_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not articles or not all(article.get("synthetic_only") is True for article in articles):
        raise ValueError("Knowledge assets must be explicitly synthetic_only")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([article["content"] for article in articles])
    evaluation = pd.read_csv(evaluation_path)
    hits = 0
    for row in evaluation.itertuples(index=False):
        scores = (vectorizer.transform([row.question]) @ matrix.T).toarray().ravel()
        retrieved = articles[int(scores.argmax())]["source_id"]
        hits += int(retrieved == row.expected_source_id)
    safety_path = Path(evaluation_path).with_name("safety_evaluation.csv")
    safety = pd.read_csv(safety_path)
    if not bool(safety["synthetic_only"].all()):
        raise ValueError("Safety evaluation data must be marked synthetic_only")
    safety_passes = 0
    safety_rows = []
    for row in safety.itertuples(index=False):
        response = safety_response(row.prompt) or ""
        required = [term.strip().lower() for term in row.required_terms.split(",")]
        passed = all(term in response.lower() for term in required)
        safety_passes += int(passed)
        safety_rows.append({"prompt": row.prompt, "category": row.category, "passed": passed, "response": response})
    metrics = {
        "documents": len(articles), "evaluation_queries": len(evaluation),
        "retrieval_recall_at_1": round(hits / max(len(evaluation), 1), 4),
        "safety_cases": len(safety), "safety_pass_rate": round(safety_passes / max(len(safety), 1), 4), "synthetic_only": True,
    }
    output = ensure_output_dir(output_dir)
    with (output / "tfidf_retriever.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "articles": articles}, handle)
    write_json(output / "retrieval_evaluation_report.json", metrics)
    pd.DataFrame(safety_rows).to_csv(output / "safety_evaluation_report.csv", index=False)
    write_model_card(
        output, "Conversational AI", "TF-IDF retrieval baseline for RAG-ready synthetic knowledge", "synthetic-1.0.0",
        knowledge_path, ["question text", "document content"], metrics,
        ["This is a synthetic retrieval baseline, not a production customer-support corpus.", "A bank-approved knowledge base, access controls, citation checks, and safety evaluation are required before LLM grounding.", "The retriever provides information only and must not direct autonomous banking actions."],
        "Retrieve relevant cited synthetic policy/product content for a controlled RAG response layer.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-path", default="data/synthetic/conversational/knowledge_base.jsonl")
    parser.add_argument("--evaluation-path", default="data/synthetic/conversational/retrieval_evaluation.csv")
    parser.add_argument("--output-dir", default="agents/conversational_ai/models")
    args = parser.parse_args()
    print(build(args.knowledge_path, args.evaluation_path, args.output_dir))
