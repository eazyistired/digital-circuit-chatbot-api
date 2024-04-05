from .qa_flow import get_qa_chain
from .llm_select import get_tokenizer_and_model as gtm
from langchain.chains import LLMChain

# FIXME Think of a better way to implement this
# This can be replaced by switch statements but I don't think it would be an improvement
# Maybe we can incorporate some of this information in the config files
model_names_vs_paths = {
    'llama-2-7b': '/'
}


DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

def get_tokenizer_and_model(model_name, quantization_config):
    return gtm(
        model_name=model_name,
        quantization_config=quantization_config
    )

def ask_question(query, llm, retriever, chat_history):
    # qa_chain = get_qa_chain(
    #     llm=llm,
    #     retriever=retriever
    # )

    prompt='Answer the question in english'
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    result = qa_chain.run(query)

    chat_history.append((query, result))

    return result, chat_history