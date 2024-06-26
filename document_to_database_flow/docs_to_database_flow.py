from .docs_to_text_chunks import get_text_chunks_from_docs
from .pdf_to_docs import get_docs
from .text_chunks_to_database import load_vector_database, store_vector_database


def convert_text_chunks_to_database(
    text_chunks, embedding_model, persist_dir, persist=True
):
    store_vector_database(text_chunks, embedding_model, persist_dir, persist)

    vector_database = load_vector_database(persist_dir, embedding_model)

    return vector_database


def convert_docs_to_database(
    documents_database_path, vector_database_path, embedding_model
):
    docs = get_docs(documents_database_path)

    # print(f'Docs: {docs}')
    if docs == []:
        return

    text_chunks = get_text_chunks_from_docs(docs)

    embedding_model = embedding_model
    vector_database = convert_text_chunks_to_database(
        text_chunks, embedding_model, vector_database_path
    )

    return vector_database


def load_database(persist_dir, embedding_model):
    embedding_model = embedding_model
    vector_database = load_vector_database(persist_dir, embedding_model)

    return vector_database


def get_retriever(vector_database, search_kwargs={"k": 3}):
    return vector_database.as_retriever(search_kwargs=search_kwargs)
