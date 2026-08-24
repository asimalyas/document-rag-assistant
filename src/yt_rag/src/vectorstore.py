import os
import uuid
import numpy as np
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedding import EmbeddingManager


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
            embedding_model: HuggingFace model name for sentence embeddings
        """
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "vector_store",
            )
        self.persist_directory = persist_directory
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(
                f"Existing documents in collection: {self.collection.count()}"
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def chunk_documents(
        self, documents, chunk_size: int = 500, chunk_overlap: int = 100
    ):
        """
        Split documents into smaller chunks while preserving metadata.

        Args:
            documents: List of LangChain Document objects
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def add_documents(self, documents: List[Any], embeddings: np.ndarray = None):
        """
        Add documents and their embeddings to the vector store.
        If embeddings is None, will generate them automatically.

        Args:
            documents: List of LangChain documents
            embeddings: Optional pre-computed embeddings
        """
        if embeddings is None:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_manager.generate_embeddings(texts)

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            # ChromaDB requires metadata values to be str, int, float, or bool
            for key, value in list(metadata.items()):
                if value is None:
                    metadata[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            # ChromaDB has batch size limits, add in batches of 500
            batch_size = 500
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end],
                )
            print(
                f"Successfully added {len(documents)} documents to vector store"
            )
            print(
                f"Total documents in collection: {self.collection.count()}"
            )
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def build_from_documents(
        self, documents, chunk_size: int = 500, chunk_overlap: int = 100
    ):
        """
        Full pipeline: chunk documents, generate embeddings, and add to store.

        Args:
            documents: List of LangChain Document objects
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunks that were added
        """
        print(f"Building vector store from {len(documents)} documents...")
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        print(f"Created {len(chunks)} chunks")

        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.add_documents(chunks, embeddings)

        return chunks

    def query(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = -100,
        doc_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score
            doc_filter: Optional source_file name to filter by

        Returns:
            List of dicts with id, content, metadata, similarity_score, distance, rank
        """
        query_embedding = self.embedding_manager.embed_query(query)

        try:
            where_filter = None
            if doc_filter:
                where_filter = {"source_file": doc_filter}

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter,
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def get_document_names(self) -> List[str]:
        """Return unique source filenames in the index"""
        try:
            all_meta = self.collection.get(include=["metadatas"])
            names = set()
            for meta in all_meta["metadatas"]:
                name = meta.get("source_file", meta.get("source", ""))
                if name:
                    names.add(name)
            return sorted(names)
        except Exception:
            return []

    def get_collection_count(self) -> int:
        """Return total number of documents in the collection"""
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"},
            )
            print("Vector store reset successfully")
        except Exception as e:
            print(f"Error resetting vector store: {e}")


# Backward-compatible alias so existing `from src.vectorstore import FaissVectorStore` works
FaissVectorStore = VectorStore
