import os
import transformers
from qa_flow.qr_handler import get_tokenizer_and_model
from qa_flow.qa_flow import get_llm_pipeline
from evaluation.evaluation import evaluate_with_ragas
from ragas.metrics import answer_similarity, answer_correctness
import pandas as pd
from datasets import Dataset
from document_to_database_flow.docs_embedding import get_embedding_model
import pickle
import json
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from qa_flow.llm_prompt import get_prompt
from langchain.chains import RetrievalQA
from document_to_database_flow.docs_to_database_flow import (
    convert_docs_to_database,
    get_retriever,
    load_database,
)

if __name__ == "__main__":
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    llm_model_path = os.path.join(project_dir_path, "models", "llama-2-13b-chat-hf")
    embedding_model_path = os.path.join(project_dir_path, "models", "instructor-xl")
    vector_database_path = os.path.join(project_dir_path, "database", "vector_store")

    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )

    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)
    embedding_model = get_embedding_model(model_path=embedding_model_path)

    vector_database = load_database(
        persist_dir=vector_database_path, embedding_model=embedding_model
    )

    retriever = get_retriever(vector_database=vector_database)
    prompt = get_prompt(prompt_selection="rag-prompt")

    rag_eval_dataset = None
    with open(
        os.path.join(script_dir_path, "testcase_results.pickle"),
        "rb",
    ) as file:
        rag_eval_dataset = pickle.load(file)

    # rag_eval_dataset = Dataset.from_pandas(pd.DataFrame(rag_eval_dataset.to_dict()))

    print(rag_eval_dataset)
    print(f"\n\n{json.dumps(rag_eval_dataset.to_dict(), indent=4)}\n\n")

    testcase_evaluation_results_df = evaluate_with_ragas(
        dataset=rag_eval_dataset,
        metrics=[answer_similarity],
        embedding_model=embedding_model,
        llm_model=llm_pipeline,
    )

    print(f"\n\n{testcase_evaluation_results_df}\n\n")
