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

if __name__ == "__main__":
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    llm_model_path = os.path.join(project_dir_path, "models", "llama-2-7b-hf")
    embedding_model_path = os.path.join(project_dir_path, "models", "instructor-xl")

    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )

    text_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        return_full_text=True,
        temperature=0.1,
        # top_p=0.95,
        # repetition_penalty=1.15,
        repetition_penalty=1.1,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline)
    embedding_model = get_embedding_model(model_path=embedding_model_path)

    with open("testcase_results.pickle", "rb") as file:
        rag_eval_dataset = pickle.load(file)

    print(f"\n\n{json.dumps(rag_eval_dataset.to_dict(), indent=4)}\n\n")
    print(embedding_model)
    print(tokenizer)
    print(model)
    print(llm_pipeline.pipeline)

    testcase_evaluation_results_df = evaluate_with_ragas(
        dataset=rag_eval_dataset,
        metrics=[answer_similarity],
        embedding_model=embedding_model,
        llm_model=llm_pipeline,
    )

    print(f"\n\n{testcase_evaluation_results_df}\n\n")
