import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

import pinecone

from documentation_helper.globals import PINECONE_INDEX_NAME

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"), environment="us-west1-gcp-free"
)


def run_llm(query: str, chat_history: list[tuple[str, Any]] = None):
    embeddings = OpenAIEmbeddings()
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    # stuff chain type only means to get the context and insert (stuff) it as the prompt context
    # the retriever is just a wrapper around the vector_store that allows us to retrieve similar vectors
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is a LangChain chain?"))
