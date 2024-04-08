from document_to_database_flow.docs_to_database_flow import (
    convert_docs_to_database,
    get_retriever,
    load_database,
)
from qa_flow.qr_handler import get_tokenizer_and_model, ask_question
import transformers
import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from qa_flow.llm_prompt import get_prompt, get_system_template
from qa_flow.qa_flow import get_llm_pipeline

if __name__ == "__main__":
    documents_database_path = "../database/documents"
    vector_database_path = "../database/vector_store"
    embedding_model_name = "instructor-xl"
    llm_model_name = "llama-2-7b-hf"
    llm_model_path = ""
    # quantization_config = None
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
    chat_history = []

    # vector_database = convert_docs_to_database(
    #     documents_database_path=documents_database_path,
    #     vector_database_path=vector_database_path,
    #     embedding_model_name=embedding_model_name
    # )

    vector_database = load_database(
        persist_dir=vector_database_path, embedding_model_name=embedding_model_name
    )

    retriever = get_retriever(vector_database=vector_database)

    tokenizer, model = get_tokenizer_and_model(
        model_name=llm_model_name, quantization_config=quantization_config
    )

    query = input("Ask your question: ")

    prompt = get_prompt(get_system_template())
    llm_pipeline = get_llm_pipeline(model=model, tokenizer=tokenizer)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
    )

    result = qa_chain(query)

    print(f"Question: {result['query']} \n")
    print(f"Answer: {result['result']} \n")
    print(f"Source Documents: {result['source_documents'][0].page_content} \n")
