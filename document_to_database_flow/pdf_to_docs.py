from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def get_docs(database_path):
    loader = DirectoryLoader(database_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    return docs
