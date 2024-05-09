import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

# if test_on_question_dataset:
#     testing_dataset = testing_dataset.add_column("answer", answers)
#     embedding_model = get_embedding_model(model_path=embedding_model_path)
#     testing_llm = LangchainLLMWrapper(model)

#     print(f"Testing dataset: {testing_dataset}\n\n")


#     result = test_llm_on_dataset(
#         testing_dataset=testing_dataset,
#         llm=testing_llm,
#         embedding_model=embedding_model,
#     )
#     df = result.to_pandas()
#     print(f"Result for testing dataset: {df.head()}")
#     df.to_csv("test_dataset_results.csv", sep=",", encoding="utf-8")
def form_dataset_from_dict(data_samples):
    return Dataset.from_dict(data_samples)


def get_testing_dataset(questions_database_path):
    df = get_question_list(questions_database_path=questions_database_path)

    data_samples = {
        "question": df["Question"],
        "contexts": list([list(x) for x in df["Context"]]),
        "ground_truth": df["Answer"],
    }

    return form_dataset_from_dict(data_samples=data_samples)


def test_llm_on_dataset(
    testing_dataset,
    llm=None,
    embedding_model=None,
):
    result = evaluate(
        dataset=testing_dataset,
        metrics=[answer_relevancy],
        llm=llm,
        embeddings=embedding_model,
        raise_exceptions=True,
    )

    return result


def get_question_list(questions_database_path):
    df = pd.read_excel(questions_database_path + "/" + "questions.xlsx")
    return df
