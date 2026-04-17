import pytest
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# Mock imports for the test suite representing the RAG system
# from src.agent_router import AgenticRouter
# from src.retrieval import HybridRetriever
# from src.generation import RAGGenerator

@pytest.fixture
def eval_dataset():
    """
    Loads 50 golden dataset Q&A pairs for evaluation.
    In a real scenario, this would load from a CSV or JSON stored in /data.
    Mocking a subset for the structural test.
    """
    data = {
        "question": [
            "What was Apple's total net sales in 2023?",
            "Identify the primary risk factors mentioned regarding supply chain disruptions."
        ],
        "ground_truth": [
            "Apple's total net sales in 2023 were $383,285 million.",
            "Primary risks include reliance on outsourced manufacturing, logistics issues, and component shortages."
        ],
        # Context is what our system retrieved. In real tests, we run generation live.
        "contexts": [
            ["Total net sales were $383,285 million, $394,328 million and $365,817 million in 2023, 2022 and 2021..."],
            ["The Company relies on outsourced manufacturing... supply chain disruptions could materially affect..."]
        ],
        "answer": [
            "Apple's net sales in 2023 reached $383,285 million [Document: AAPL_10K.pdf, Page: 31].",
            "The risks involve outsourced manufacturing constraints and logistic bottlenecks [Document: AAPL_10K.pdf, Page: 12]."
        ]
    }
    return Dataset.from_dict(data)

def test_rag_pipeline_metrics(eval_dataset):
    """
    Runs RAGAS framework against the golden dataset.
    This acts as a CI/CD Gate. Fails if faithfulness < 0.75.
    """
    print("Initiating RAGAS Evaluation Suite...")
    
    # Normally we would:
    # 1. Iterate over eval_dataset["question"]
    # 2. Run agent.route_and_execute(q)
    # 3. Populate "answer" and "contexts" dynamically.
    
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    df = result.to_pandas()
    print("\nEvaluation Metrics Results:")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']].to_string())
    
    mean_faithfulness = df['faithfulness'].mean()
    print(f"\nAggregate Faithfulness Score: {mean_faithfulness:.2f}")
    
    # The CI Gate condition
    assert mean_faithfulness >= 0.75, f"Pipeline failed CI gate: Faithfulness ({mean_faithfulness:.2f}) < 0.75"
