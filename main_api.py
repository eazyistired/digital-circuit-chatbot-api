from document_to_database_flow.docs_to_database_flow import convert_docs_to_database, get_retriever, load_database
from qa_flow.qr_handler import get_tokenizer_and_model, get_chain, ask_question
import transformers
import torch

if __name__ == '__main__':
    documents_database_path = '../database/documents'
    vector_database_path = '../database/vector_store'
    embedding_model_name = 'instructor-xl'
    llm_model_name = 'llama-2-7b-hf'
    llm_model_path = ''
    # quantization_config = None
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=getattr(torch, 'float16'),
        bnb_4bit_use_double_quant=False
    )
    chat_history = []

    # vector_database = convert_docs_to_database(
    #     documents_database_path=documents_database_path,
    #     vector_database_path=vector_database_path,
    #     embedding_model_name=embedding_model_name
    # )

    vector_database = load_database(
        persist_dir=vector_database_path,
        embedding_model_name=embedding_model_name
    )

    retriever = get_retriever(
        vector_database=vector_database
    )

    tokenizer, model = get_tokenizer_and_model(
        model_name=llm_model_name,
        quantization_config=quantization_config
    )

    # chain = get_chain(
    #     llm=model,
    #     retriever=None,
    #     chain_type='stuff',
    #     return_source_documents=True
    # )

    query = input('Ask your question: ')

    # result, chat_history = ask_question(
    #     query=query,
    #     chain=chain,
    #     chat_history=chat_history
    # )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=200
    )

    result = pipeline(f'<s>[INST] Please answer the question in english: {query} [/INST]')

    print('Answer: ' + result[0]['generated_text'] + '\n')