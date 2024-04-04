from .document_to_database_flow.docs_to_database_flow import convert_docs_to_database, get_retriever
from .qa_flow.qr_handler import get_tokenizer_and_model, get_chain, ask_question

if __name__ == '__main__':
    database_path = '../database'
    embedding_model_name = 'instructor-xl'
    model_name = 'llama-2-7b'
    quantization_config = None
    chat_history = []

    vector_database = convert_docs_to_database(
        database_path=database_path,
        embedding_model_name=embedding_model_name
    )

    retriever = get_retriever(
        vector_database=vector_database
    )

    tokenizer, model = get_tokenizer_and_model(
        model_name=model_name,
        quantization_config=quantization_config
    )

    chain = get_chain(
        llm=model,
        retriever=None,
        chain_type='stuff',
        return_source_documents=True
    )

    query = input('Ask your question: ')

    result, chat_history = ask_question(
        query=query,
        chain=chain,
        chat_history=chat_history
    )

    print('Answer: ' + result['answer'] + '\n')