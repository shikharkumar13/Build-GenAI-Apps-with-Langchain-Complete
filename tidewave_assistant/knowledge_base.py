from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documents import load_directory


def build_retriever(
    docs_path: str = "tidewave_docs",
    embedding: Embeddings | None = None,
    persist_directory: str | None = None,
    k: int = 3,
) -> BaseRetriever:
    embedding = embedding or OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="tidewave_docs",
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    if not vectorstore.get(limit=1)["ids"]:  # empty: first run, or in-memory
        docs = load_directory(docs_path)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        ).split_documents(docs)
        vectorstore.add_documents(chunks)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)
