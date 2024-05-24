from document_to_database_flow.docs_to_database_flow import (
    convert_docs_to_database,
    get_retriever,
    load_database,
)
from testing_qa import get_testing_dataset
from qa_flow.qr_handler import get_tokenizer_and_model
import transformers
import torch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from qa_flow.llm_prompt import get_prompt
from qa_flow.qa_flow import get_llm_pipeline
import json
import os
import pandas as pd
from document_to_database_flow.docs_embedding import get_embedding_model
from ragas.llms import LangchainLLMWrapper
from evaluation.testing import test_and_evaluate_on_dataset, save_results


def load_config_object(config_object_path):
    with open(config_object_path) as config_file:
        return json.load(config_file)


CONFIG_TYPE = "development"


def get_qa_chain():
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    config_object_path = os.path.join(script_dir_path, "config.json")
    configs = load_config_object(config_object_path=config_object_path)
    config = configs[CONFIG_TYPE]

    embedding_model_name = config["embedding_model_name"]
    llm_model_name = config["llm_model_name"]
    convert_to_database = config["convert_docs_to_database"]
    prompt_selection = config["prompt_selection"]
    ask_your_own_questions = config["ask_your_own_questions"]

    print(f"\n\nConfig object: {json.dumps(config, indent=4)}\n\n")

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    chat_history = []

    # PATH CONFIGS
    documents_database_path = os.path.join(project_dir_path, "database", "documents")
    vector_database_path = os.path.join(project_dir_path, "database", "vector_store")
    questions_database_path = os.path.join(project_dir_path, "database", "questions")
    embedding_model_path = os.path.join(
        project_dir_path, "models", embedding_model_name
    )
    testcases_results_folder_path = os.path.join(
        project_dir_path, "database", "results"
    )
    llm_model_path = os.path.join(project_dir_path, "models", llm_model_name)

    # CODE
    embedding_model = get_embedding_model(model_path=embedding_model_path)

    if convert_to_database:
        convert_docs_to_database(
            documents_database_path=documents_database_path,
            vector_database_path=vector_database_path,
            embedding_model=embedding_model,
        )

    vector_database = load_database(
        persist_dir=vector_database_path, embedding_model=embedding_model
    )

    retriever = get_retriever(vector_database=vector_database)

    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )

    prompt = get_prompt(prompt_selection=prompt_selection)
    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )

    return qa_chain


def generate_response(qa_chain, query):
    return qa_chain.invoke(query)


if __name__ == "__main__":
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    config_object_path = os.path.join(script_dir_path, "config.json")
    configs = load_config_object(config_object_path=config_object_path)
    config = configs[CONFIG_TYPE]

    embedding_model_name = config["embedding_model_name"]
    llm_model_name = config["llm_model_name"]
    convert_to_database = config["convert_docs_to_database"]
    prompt_selection = config["prompt_selection"]
    ask_your_own_questions = config["ask_your_own_questions"]

    print(f"\n\nConfig object: {json.dumps(config, indent=4)}\n\n")

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    chat_history = []

    # PATH CONFIGS
    documents_database_path = os.path.join(project_dir_path, "database", "documents")
    vector_database_path = os.path.join(project_dir_path, "database", "vector_store")
    questions_database_path = os.path.join(project_dir_path, "database", "questions")
    embedding_model_path = os.path.join(
        project_dir_path, "models", embedding_model_name
    )
    testcases_results_folder_path = os.path.join(
        project_dir_path, "database", "results"
    )
    llm_model_path = os.path.join(project_dir_path, "models", llm_model_name)

    # CODE
    embedding_model = get_embedding_model(model_path=embedding_model_path)

    if convert_to_database:
        convert_docs_to_database(
            documents_database_path=documents_database_path,
            vector_database_path=vector_database_path,
            embedding_model=embedding_model,
        )

    vector_database = load_database(
        persist_dir=vector_database_path, embedding_model=embedding_model
    )

    retriever = get_retriever(vector_database=vector_database)

    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )

    prompt = get_prompt(prompt_selection=prompt_selection)
    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )

    testing_dataset = get_testing_dataset(
        questions_database_path=questions_database_path
    )

    while ask_your_own_questions:
        question = input("Ask your question:")

        result = qa_chain.invoke(question)

        print(
            "\n\n ===================================================================== \n\n"
        )

    testcase_results_list = test_and_evaluate_on_dataset(
        qa_chain=qa_chain,
        embedding_model=embedding_model,
        test_dataset=testing_dataset,
    )

    save_results(
        results_list=testcase_results_list,
        config_object=config,
        testcases_results_folder_path=testcases_results_folder_path,
    )
