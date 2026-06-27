import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Sample Golden Set (You should expand this to 50-100 Q&A pairs as per the assignment)
data = {
    "question": [
        "Who wrote Hind Swaraj?",
        "What was the Rowlatt Act?",
        "Why did nationalism rise in 19th-century Europe?"
    ],
    "answer": [
        "Mahatma Gandhi wrote Hind Swaraj in 1909.",
        "The Rowlatt Act of 1919 gave the government enormous powers to repress political activities and allowed detention of political prisoners without trial for two years.",
        "Nationalism rose in 19th-century Europe due to a combination of political, social, and economic factors including the ideas of the French Revolution and the rise of a new middle class."
    ],
    "contexts": [
        ["Mahatma Gandhi declared in his book Hind Swaraj (1909) that British rule was established in India with the cooperation of Indians."],
        ["The Rowlatt Act (1919) had been hurriedly passed through the Imperial Legislative Council... It gave the government enormous powers to repress political activities."],
        ["The first clear expression of nationalism came with the French Revolution in 1789... The growth of industrial production and trade meant the growth of towns and the emergence of commercial classes."]
    ],
    "ground_truth": [
        "Mahatma Gandhi",
        "An act that gave the government powers to repress political activities.",
        "French Revolution, middle class emergence, and industrialization."
    ]
}

dataset = Dataset.from_dict(data)

def run_evaluation():
    print("Running RAGAS Evaluation on Golden Set...")
    # NOTE: Ragas requires OPENAI_API_KEY by default for evaluation models (LLM-as-a-judge)
    # Ensure OPENAI_API_KEY is in your environment variables.
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
    )
    
    print("Evaluation Results:")
    print(result)

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: Ragas typically requires an OPENAI_API_KEY for the judge LLM.")
    run_evaluation()
