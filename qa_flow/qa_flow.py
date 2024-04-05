from langchain.chains import ConversationalRetrievalChain, RetrievalQA

def get_qa_chain(llm, retriever, chain_type='stuff', return_source_documents=True):
    # return ConversationalRetrievalChain.from_llm(llm=llm, 
    #                             #   chain_type=chain_type, 
    #                               retriever=retriever, 
    #                               return_source_documents=return_source_documents)

  RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever
  )