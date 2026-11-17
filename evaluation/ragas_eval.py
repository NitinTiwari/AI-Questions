"""
RAG evaluation using RAGAS.

Reads a set of test questions (with optional ground-truth answers) from
evaluation/testset.json, runs them through the live agent to collect the
answer + retrieved contexts for each, scores them with RAGAS metrics, and
writes both per-question and summary results to CSV.

Usage:
    python -m evaluation.ragas_eval
    python -m evaluation.ragas_eval --testset evaluation/testset.json
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset

# pyrefly: ignore [missing-import]
from ragas import evaluate
# pyrefly: ignore [missing-import]
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.agent import build_agent
from src.tools import search_document

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TESTSET = Path(__file__).resolve().parent / "testset.json"


def load_testset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_agent_over_testset(testset: list[dict]) -> dict:
    """Collect question / answer / retrieved-contexts / ground_truth columns."""
    agent_executor = build_agent()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in testset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        # Pull the raw retrieved passages directly from the retriever tool so
        # RAGAS can score groundedness against exactly what the agent saw.
        raw_context = search_document.invoke({"query": question})
        context_list = [p.strip() for p in raw_context.split("\n\n") if p.strip()]

        result = agent_executor.invoke({"input": question})
        answer = result["output"]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def evaluate_rag(testset_path: Path = DEFAULT_TESTSET) -> Path:
    settings.validate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    testset = load_testset(testset_path)
    data = run_agent_over_testset(testset)
    dataset = Dataset.from_dict(data)

    # RAGAS needs its own LLM (for judgment) and embeddings (for similarity metrics).
    judge_llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    per_question_df = scores.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_path = RESULTS_DIR / f"ragas_eval_{timestamp}.csv"
    per_question_df.to_csv(detailed_path, index=False)

    summary_path = RESULTS_DIR / f"ragas_eval_{timestamp}_summary.csv"
    summary_df = pd.DataFrame([{
        metric: per_question_df[metric].mean()
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if metric in per_question_df.columns
    }])
    summary_df.to_csv(summary_path, index=False)

    print(f"Detailed per-question results: {detailed_path}")
    print(f"Summary (mean scores):        {summary_path}")
    print(summary_df.to_string(index=False))

    return detailed_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG app with RAGAS")
    parser.add_argument(
        "--testset",
        type=Path,
        default=DEFAULT_TESTSET,
        help="Path to a JSON file with a list of {question, ground_truth} objects",
    )
    args = parser.parse_args()
    evaluate_rag(args.testset)
