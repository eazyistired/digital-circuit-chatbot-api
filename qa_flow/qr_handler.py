from .qa_flow import get_qa_chain, get_llm_pipeline
from .llm_prompt import get_prompt, get_system_template
from .llm_select import get_tokenizer_and_model as gtm
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

# FIXME Think of a better way to implement this
# This can be replaced by switch statements but I don't think it would be an improvement
# Maybe we can incorporate some of this information in the config files
model_names_vs_paths = {"llama-2-7b": "/"}


def get_tokenizer_and_model(model_name, quantization_config):
    return gtm(model_name=model_name, quantization_config=quantization_config)


def _ask_question(query, llm, retriever, chat_history):
    prompt = "Answer the question in english"
    qa_chain = LLMChain(llm=llm, prompt=prompt)

    result = qa_chain.run(query)

    chat_history.append((query, result))

    return result, chat_history


def ask_question(qa_chain, question):
    result = qa_chain(question)

    return result
