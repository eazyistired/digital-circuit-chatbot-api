import numpy as np
from ragas import evaluate


# Measuring the semantic similarity between answer and ground_truth
def get_answer_similarity(answer_embeddings, ground_truth_embeddings) -> list:
    # FIXME Find out if here it's ok to pass a list of answers or one answer at a time
    answer_embeddings = np.array(answer_embeddings)
    ground_truth_embeddings = np.array(ground_truth_embeddings)

    dot_product = np.dot(answer_embeddings, ground_truth_embeddings)
    magnitude_answer_embeddings = np.linalg.norm(answer_embeddings)
    magnitude_ground_truth_embeddings = np.linalg.norm(ground_truth_embeddings)

    cosine_similarity = dot_product / (
        magnitude_answer_embeddings * magnitude_ground_truth_embeddings
    )

    return cosine_similarity


def evaluate_with_ragas(dataset, metrics: list, embedding_model, llm_model):
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=True,
        embeddings=embedding_model,
        llm=llm_model,
    )

    return result
