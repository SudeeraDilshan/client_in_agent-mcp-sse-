import os

from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def create_vector_store() -> FAISS:
    """Create a vector store from the knowledge base text file."""
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, "base.txt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.from_documents(texts, embeddings)

        return vector_store

    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise RuntimeError(f"Failed to load knowledge base: {str(e)}")


def create_retriever_tool_func():
    vb = create_vector_store()
    retriever = vb.as_retriever()
    return create_retriever_tool(retriever=retriever, name="search_about_John",
                                 description="Search for information about John Anderson")


ask_about_John = create_retriever_tool_func()

rag_tool = [ask_about_John]
