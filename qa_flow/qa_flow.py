from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import transformers


def get_qa_chain(llm, retriever, chain_type="stuff", return_source_documents=True):
    # return ConversationalRetrievalChain.from_llm(llm=llm,
    #                             #   chain_type=chain_type,
    #                               retriever=retriever,
    #                               return_source_documents=return_source_documents)

    RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def get_llm_pipeline(
    model,
    tokenizer,
):
    # streamer = transformers.TextStreamer(
    #     # FIXME Is this skip_prompt an issue?
    #     tokenizer,
    #     skip_prompt=True,
    #     skip_special_tokens=True,
    # )
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        # temperature=0.001,
        # top_p=0.95,
        repetition_penalty=1.15,
        # streamer=streamer,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline)

    return llm_pipeline
