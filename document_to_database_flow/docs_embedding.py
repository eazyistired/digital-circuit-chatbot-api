from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
)
from sentence_transformers import SentenceTransformer

# FIXME Think of a better way to implement this
# This can be replaced by switch statements but I don't think it would be an improvement
# Maybe we can incorporate some of this information in the config files

model_names_vs_classes = {
    "instructor-xl": HuggingFaceInstructEmbeddings,
    "e5-mistral-7b-instruct": HuggingFaceEmbeddings,
}


def get_embedding_model_from_path(model_path, model_class):
    return model_class(
        model_name=model_path,
        model_kwargs={"device": "cuda"},
    )


def get_embedding_model(model_path, model_name):
    return get_embedding_model_from_path(
        model_path=model_path, model_class=model_names_vs_classes[model_name]
    )
