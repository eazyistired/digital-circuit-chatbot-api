from langchain_core.prompts import PromptTemplate


def generate_prompt(system_prompt: str, prompt: str) -> str:
    return f"""
    <s>[INST] <<SYS>>
        {system_prompt}
    <</SYS>>

        {prompt} [/INST]
    """.strip()


def get_template(system_prompt):
    template = generate_prompt(
        system_prompt=system_prompt,
        prompt="""
        Context: {context}

        Chat history: {ceva}

        Question: {question}

        Answer:
        """.strip(),
    )

    return template


# TODO Make this dynamic. So you can have a way to enter your own prompt and save it or smth
def get_system_prompt(prompt_selection):
    match prompt_selection:
        case "system-prompt":
            return "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        case "rag-prompt":
            return """
                Using the information contained in the context, give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant to the question.
                If the answer cannot be deduced from the context, do not give an answer.
            """.strip()

        case _:
            return """
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
            """.strip()


def get_prompt(prompt_selection):
    system_prompt = get_system_prompt(prompt_selection=prompt_selection)
    template = get_template(system_prompt=system_prompt)

    return PromptTemplate(
        template=template,
        input_variables=["ceva", "question", "context"],
    )
