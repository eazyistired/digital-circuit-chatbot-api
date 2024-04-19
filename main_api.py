from document_to_database_flow.docs_to_database_flow import (
    convert_docs_to_database,
    get_retriever,
    load_database,
)
from testing_qa import get_question_list
from qa_flow.qr_handler import get_tokenizer_and_model, ask_question
import transformers
import torch
from langchain.chains import RetrievalQA
from qa_flow.llm_prompt import get_prompt
from qa_flow.qa_flow import get_llm_pipeline
import json


def load_config_object(config_object_path):
    with open(config_object_path) as config_file:
        return json.load(config_file)


if __name__ == "__main__":
    config_object_path = "config.json"
    configs = load_config_object(config_object_path=config_object_path)
    config = configs["development"]

    documents_database_path = config["documents_database_path"]
    vector_database_path = config["vector_database_path"]
    questions_database_path = config["questions_database_path"]
    embedding_model_name = config["embedding_model_name"]
    llm_model_name = config["llm_model_name"]
    convert_to_database = config["convert_docs_to_database"]
    prompt_selection = config["prompt_selection"]

    print(f"\n\nConfig object: {json.dumps(config, indent=4)}\n\n")

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    chat_history = []

    if convert_to_database:
        convert_docs_to_database(
            documents_database_path=documents_database_path,
            vector_database_path=vector_database_path,
            embedding_model_name=embedding_model_name,
        )

    vector_database = load_database(
        persist_dir=vector_database_path, embedding_model_name=embedding_model_name
    )

    retriever = get_retriever(vector_database=vector_database)

    tokenizer, model = get_tokenizer_and_model(
        model_name=llm_model_name, quantization_config=quantization_config
    )

    prompt = get_prompt(prompt_selection=prompt_selection)
    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )

    question_list = get_question_list(questions_database_path=questions_database_path)
    for question in question_list:
        result = qa_chain(question)

        # print(f"Question: {result['query']} \n")
        print(f"Answer: {result['result']} \n \n \n")
        print(
            f"=================================================================================="
        )
        print(
            f"=================================================================================="
        )
        # print(f"Source Documents: {result['source_documents'][0].page_content} \n")
