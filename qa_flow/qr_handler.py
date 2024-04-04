from qa_flow import get_qa_chain
from llm_select import get_tokenizer_and_model

# FIXME Think of a better way to implement this
# This can be replaced by switch statements but I don't think it would be an improvement
# Maybe we can incorporate some of this information in the config files
model_names_vs_paths = {
    'llama-2-7b': '/'
}

def get_question_result(query, chain, chat_history):
    return chain({'question': query, 'chat_history': chat_history})

def get_chain(llm, retriever, chain_type, return_source_documents):
    return get_qa_chain(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=return_source_documents
    )

def get_tokenizer_and_model(model_name, quantization_config):
    return get_tokenizer_and_model(
        model_name=model_name,
        model_path=model_names_vs_paths[model_name],
        quantization_config=quantization_config
    )

def ask_question(query, chain, chat_history):
    result = get_question_result(
        query=query,
        qa_chain=chain,
        chat_history=chat_history
    )
    chat_history.append((query, result['answear']))

    return result, chat_history