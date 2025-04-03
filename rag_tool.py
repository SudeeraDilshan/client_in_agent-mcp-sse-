import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class RAGTool:
    """
    A tool for Retrieval-Augmented Generation (RAG) that provides relevant context
    from a knowledge base for answering user queries.
    """

    def __init__(self):
        """Initialize the RAG tool with a vector store from the knowledge base."""
        self.vector_store = self._create_vector_store()

    def _create_vector_store(self):
        """Create a vector store from the knowledge base text file."""
        try:
            # Dynamically construct the file path
            current_directory = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_directory, "knowledge_base.txt")

            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load the text file
            loader = TextLoader(file_path)
            documents = loader.load()

            # Split the text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(texts, embeddings)

            return vector_store

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise RuntimeError(f"Failed to load knowledge base: {str(e)}")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        try:
            # Create a retriever from the vector store
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

            # Get relevant documents
            docs = retriever.get_relevant_documents(query)

            # Format the documents into a string
            context = self._format_docs(docs)

            return context

        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return "No relevant information found in the knowledge base."

    def _format_docs(self, docs: List[Document]) -> str:
        """Format the documents into a string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _run(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            context = self.get_relevant_context(query, k)
            return {
                "status": "success",
                "context": context
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "context": "Failed to retrieve relevant information."
            }