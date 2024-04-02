from langchain.embeddings import HuggingFaceInstructEmbeddings

# TODO Think of a better way to implement this
# This can be replaced by switch statements but I don't think it would be an improvement
# Maybe we can incorporate some of this information in the config files
model_names_vs_paths = {
    'instructor-xl': '/'
}

model_names_vs_classes = {
    'instructor-xl': HuggingFaceInstructEmbeddings()
}

def get_embedding_model_from_path(model_path, model_class):
    return model_class(model_path, model_kwargs={"device": 'cuda'})


def get_embedding_model(model_name):
    model = get_embedding_model_from_path(model_names_vs_paths[model_name], model_names_vs_classes[model_name])
    return model