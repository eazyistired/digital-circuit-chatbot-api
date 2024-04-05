from document_to_database_flow.docs_to_database_flow import convert_docs_to_database, get_retriever, load_database
from qa_flow.qr_handler import get_tokenizer_and_model, ask_question
import transformers
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

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

if __name__ == '__main__':
    documents_database_path = '../database/documents'
    vector_database_path = '../database/vector_store'
    embedding_model_name = 'instructor-xl'
    llm_model_name = 'llama-2-7b-hf'
    llm_model_path = ''
    # quantization_config = None
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=getattr(torch, 'float16'),
        bnb_4bit_use_double_quant=False
    )
    chat_history = []

    # vector_database = convert_docs_to_database(
    #     documents_database_path=documents_database_path,
    #     vector_database_path=vector_database_path,
    #     embedding_model_name=embedding_model_name
    # )

    vector_database = load_database(
        persist_dir=vector_database_path,
        embedding_model_name=embedding_model_name
    )

    retriever = get_retriever(
        vector_database=vector_database
    )

    tokenizer, model = get_tokenizer_and_model(
        model_name=llm_model_name,
        quantization_config=quantization_config
    )

    query = input('Ask your question: ')

    # result, chat_history = ask_question(
    #     query=query,
    #     llm=model,
    #     retriever=retriever,
    #     chat_history=chat_history
    # )

    streamer = transformers.TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})

    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """
        {context}

        Question: {question}
        """,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    result = qa_chain(query)

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.float16,
    #     max_length=200
    # )

    # result = pipeline(f'<s>[INST] Please answer the question in english: {query} [/INST]')

    # print('Answer: ' + result[0]['generated_text'] + '\n')
    print(f"Question: {result['query']} \n")
    print(f"Answer: {result['result']} \n")
    print(f"Source Documents: {result['source_documents'][0].page_content} \n")