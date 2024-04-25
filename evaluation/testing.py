import pandas as pd
import time
import os
import json


def check_dir_or_mkdir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        return

    os.mkdir(path=dir_path)


def get_test_results(qa_chain, test_dataset):
    results = []
    for query in test_dataset:
        chain_result = qa_chain(query["question"])

        result = {
            "answer": chain_result[
                "result"
            ],  # FIXME Find a way to eliminate the garbage and keep only the answer
            "question": chain_result["query"],
            "contexts": [
                source_document for source_document in chain_result["source_documents"]
            ],
            # FIXME Find a way to pass the contexts here; It can be retrieved from the final result but I think directly from the retriever is better
            "ground_truth": query["ground_truth"],
        }

        results.append(result)
    return results


def store_results(
    results_list: list, config_object: dict, testcases_results_folder_path
):
    results_df = pd.DataFrame(columns=[results_list[0].keys()])
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

    # Store results to csv
    results_df.to_csv(
        path_or_buf=os.path.join(
            testcase_results_folder_path, f"results_{testcase_results_nametag}.csv"
        ),
        index=True,
        sep="|",
        encoding="utf-8",
    )


def test_on_dataset(
    qa_chain, test_dataset, config_object, testcases_results_folder_path
):
    test_results = get_test_results(qa_chain=qa_chain, test_dataset=test_dataset)
    store_results(
        results_list=test_results,
        config_object=config_object,
        testcases_results_folder_path=testcases_results_folder_path,
    )
