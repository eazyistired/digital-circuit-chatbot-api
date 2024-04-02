from langchain.text_splitter import RecursiveCharacterTextSplitter

def convert_docs_to_text(docs):
    # TODO Do some post processig of the docs, maybe?
    return docs

# FIXME add recursive text splitter and choose some relevant chunk_size parameters
def split_text_into_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    text_chunks = text_splitter.split_documents(text)
    
    return text_chunks

def get_text_chunks_from_docs(docs, chunk_size=1000, chunk_overlap=200):
    text = convert_docs_to_text(docs)
    text_chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

    return text_chunks