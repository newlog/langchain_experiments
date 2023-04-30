import os

from langchain import VectorDBQA, OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore


def start():
    initialize_pinecone()
    documents = load_text_data()
    texts = split_documents(documents)
    embeddings = create_embeddings(texts)
    vector_store = insert_to_vectordb(texts, embeddings)
    query = 'What is a vector DB? Give me a 15 word answer for a beginner'
    answer = ask_to_llm_using_vdb(vector_store, query)
    print(f'-> {query}')
    print(f'-> {answer}')


def initialize_pinecone():
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="us-west1-gcp-free")


def load_text_data() -> list[Document]:
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    documents = loader.load()  # it will only return one document
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    # chunk_size and chunk_overlap are parameters you might need to tweak for the LLM
    # to respond appropriately
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts


def create_embeddings(texts: list[Document]) -> OpenAIEmbeddings:
    # the embeddings object does not have much information because the
    # embeddings are stored in openai.
    # By default, the embeddings model used is text-embedding-ada-002.
    # This is important because Pinecone (vector db) will ask for the dimensions (euclidean) of the
    # embeddings later on, which are found in the OpenAI's documentation for that model.
    embeddings = OpenAIEmbeddings()
    return embeddings


def insert_to_vectordb(
    texts: list[Document], embeddings: OpenAIEmbeddings
) -> VectorStore:
    # This will store the text and embeddings in the remote Pinecone Vector DB.
    # Take into account that this function keeps adding more and more vectors to the index even if the texts are the same.
    # It seems like embeddings might be different even though the texts are the same. Or maybe the texts are different because
    # the splitting is not deterministic (TBD).
    vectorstore = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )
    return vectorstore


def ask_to_llm_using_vdb(vector_store: VectorStore, query: str):
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=vector_store, return_source_documents=True
    )
    result = qa({"query": query})
    print(f'LLM used {len(result["source_documents"])} documents to reason an answer')
    return result['result']


if __name__ == "__main__":
    start()
