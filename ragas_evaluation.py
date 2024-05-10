import os
import json
import sys
import ragas.metrics
import transformers
from document_to_database_flow.docs_embedding import get_embedding_model
from qa_flow.qr_handler import get_tokenizer_and_model
from qa_flow.qa_flow import get_llm_pipeline
import pandas as pd
from datasets import Dataset
import ragas


def load_config_object(config_object_path):
    with open(config_object_path) as config_file:
        return json.load(config_file)


CONFIG_TYPE = "development"

if __name__ == "__main__":
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    config_object_path = os.path.join(script_dir_path, "config.json")
    configs = load_config_object(config_object_path=config_object_path)
    config = configs[CONFIG_TYPE]

    embedding_model_name = config["embedding_model_name"]
    llm_model_name = "llama-2-13b-hf"
    prompt_selection = config["prompt_selection"]

    print(f"\n\nConfig object: {json.dumps(config, indent=4)}\n\n")

    # PATH CONFIGS
    embedding_model_path = os.path.join(
        project_dir_path, "models", embedding_model_name
    )
    llm_model_path = os.path.join(project_dir_path, "models", llm_model_name)

    # CODE
    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    embedding_model = get_embedding_model(model_path=embedding_model_path)

    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )

    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)

    testing_df = pd.read_csv(
        "/mnt/Storage1/grozavu/digital-circuit-chatbot/database/results/test_09_05_2024__18_36_16/results_test_09_05_2024__18_36_16.csv",
        sep="|",
        index_col=False,
    )

    ragas_dict = {"answer": [], "question": [], "contexts": [], "ground_truth": []}
    for el in testing_df["answer"]:
        ragas_dict["answer"].append(el)
    for el in testing_df["question"]:
        ragas_dict["question"].append(el)
    for el in testing_df["ground_truth"]:
        ragas_dict["ground_truth"].append(el)
    for el in testing_df["contexts"]:
        new_list = []
        new_list.append(el)
        ragas_dict["contexts"].append(new_list)

    # print(f"\n\n{json.dumps(testing_df.to_dict(), indent=4)}\n\n")
    # print(f"\n\n{json.dumps(ragas_dict, indent=4)}\n\n")

    ragas_evaluation_dataset = Dataset.from_dict(ragas_dict)
    print(f"\n\n{ragas_evaluation_dataset}\n\n")
    print(f"\n\n{json.dumps(ragas_evaluation_dataset.to_dict(), indent=4)}\n\n")

    result = ragas.evaluate(
        dataset=ragas_evaluation_dataset,
        metrics=[ragas.metrics.answer_correctness],
        llm=llm_pipeline,
        embeddings=embedding_model,
    )

    print(f"\n\n{result}\n\n")
