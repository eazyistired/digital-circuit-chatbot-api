import time
from langchain_community.vectorstores import Chroma

def load_vector_database(persist_dir, embedding_model):
    return Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding_model
    )

def store_vector_database(text_chunks, embedding_model, persist_dir, persist=True):
    if persist == False:
        return
    
    start_time = time.time()
    vectordb = Chroma.from_documents(
        documents=text_chunks, 
        embedding=embedding_model, 
        persist_directory=persist_dir
    )
    end_time = time.time()
    print(f"Total time it took to store vector database: {end_time-start_time}s")

    vectordb.persist()

def get_retriever(vector_database, search_kwargs={'k': 3}):
    return vector_database.as_retriever(search_kwargs=search_kwargs)