from docs_to_text_chunks import get_text_chunks_from_docs
from docs_embedding import get_embedding_model
from pdf_to_docs import get_docs
from text_chunks_to_database import load_vector_database, store_vector_database

def convert_text_chunks_to_database(text_chunks, embedding_model, persist_dir, persist=True):
    store_vector_database(
        text_chunks,
        embedding_model,
        persist_dir,
        persist
    )

    vector_database = load_vector_database(
        persist_dir,
        embedding_model
    )

    return vector_database

def convert_docs_to_database(database_path, embedding_model_name):
    docs = get_docs(database_path)
    text_chunks = get_text_chunks_from_docs(docs)

    embedding_model = get_embedding_model(embedding_model_name)
    # FIXME Add some logic to store the database or load the database in a separate function, don't do the whole logic here
    database = convert_text_chunks_to_database(
        text_chunks,
        embedding_model,
        database_path
    )

    return database