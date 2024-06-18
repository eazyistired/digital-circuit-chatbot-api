import pandas as pd
import time
import os
import json
from .evaluation import get_answer_similarity


def check_dir_or_mkdir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        return

    os.mkdir(path=dir_path)


def get_test_results(qa_chain, test_dataset):
    """
    Return type: A list of dicts. Each dict represents a query asked to the llm and has:
    - answer
    - question
    - contexts
    - ground_truth
    """

    results = []
    inference_time_periods = []
    for query in test_dataset:
        start_time = time.time()
        chain_result = qa_chain.invoke(
            {
                "question": query["question"],
                "chat_history": [],
                "ceva": "",
            },
            return_only_outputs=True,
        )
        end_time = time.time()

        inference_time_periods.append(end_time - start_time)

        delimiter = "[/INST]"
        index = chain_result["answer"].find(delimiter)
        answer = (
            chain_result["answer"][index + len(delimiter) :]
            if index != -1
            else chain_result["answer"]
        )

        result = {
            # To get the last part of the result that has only the answer
            "answer": answer,
            "question": query["question"],
            "contexts": [
                source_document for source_document in chain_result["source_documents"]
            ],
            "ground_truth": query["ground_truth"],
        }

        results.append(result)

    sum = 0
    for inference_time in inference_time_periods:
        sum += inference_time
    average_inference_time = sum / len(inference_time_periods)

    print(f"\n\n\t\t\t Average inference time: {average_inference_time}\t\t\t\n\n")
    return results, average_inference_time


def store_metric_in_results(
    results_list: list, metric_name: str, metric_scores: list
) -> list:
    assert len(results_list) == len(
        metric_scores
    ), f"Length of results list doesn't match length of metric values list for metric: {metric_name}"

    for i in range(0, len(results_list)):
        results_list[i][metric_name] = metric_scores[i]

    return results_list


def save_results(
    testcases_results_folder_path,
    results_list: list,
    config_object: dict,
    average_inference_time,
):
    testcase_results_nametag = (
        f"test_{time.strftime('%d_%m_%Y__%H_%M_%S', time.localtime())}"
    )
    testcase_results_folder_path = os.path.join(
        testcases_results_folder_path, testcase_results_nametag
    )

    # Store results in dataframe
    results_df = pd.DataFrame(results_list)

    # Crate testcase result folder
    check_dir_or_mkdir(dir_path=testcase_results_folder_path)

    # Store configuration
    with open(
        os.path.join(
            testcase_results_folder_path, f"{testcase_results_nametag}_config.json"
        ),
        "w",
    ) as file:
        json.dump(config_object, file, indent=4)
        json.dump(average_inference_time, file)

    # Store results to csv
    results_df.to_csv(
        path_or_buf=os.path.join(
            testcase_results_folder_path, f"results_{testcase_results_nametag}.csv"
        ),
        index=True,
        sep="|",
        encoding="utf-8",
    )


def test_and_evaluate_on_dataset(
    qa_chain,
    embedding_model,
    test_dataset,
):
    testcase_results_list, average_inference_time = get_test_results(
        qa_chain=qa_chain, test_dataset=test_dataset
    )

    metric_scores = evaluate_on_results(
        embedding_model=embedding_model, results_list=testcase_results_list
    )

    testcase_results_list = store_metric_in_results(
        results_list=testcase_results_list,
        metric_name="answer_similarity",
        metric_scores=metric_scores,
    )

    return testcase_results_list, average_inference_time


# FIXME Add dynamic metric calculator
def evaluate_on_results(embedding_model, results_list):
    answears_similarity = []

    for result in results_list:
        answer_embeddings = embedding_model.embed_query(result["answer"])
        ground_truth_embeddings = embedding_model.embed_query(result["ground_truth"])

        answer_similarity = get_answer_similarity(
            answer_embeddings=answer_embeddings,
            ground_truth_embeddings=ground_truth_embeddings,
        )

        answears_similarity.append(answer_similarity)

    return answears_similarity
