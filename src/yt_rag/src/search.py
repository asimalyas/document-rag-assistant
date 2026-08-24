import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .vectorstore import VectorStore

# Load .env from the yt_rag directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class RAGSearch:
    """RAG pipeline with retrieval trace, multi-turn conversation, and document scoping."""

    def __init__(
        self,
        persist_dir: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
        vectorstore: VectorStore = None,
    ):
        """
        Initialize the RAG search.

        Args:
            persist_dir: Directory for the vector store (ChromaDB)
            embedding_model: HuggingFace model name for embeddings
            llm_model: Groq model name for the LLM
            vectorstore: Optional pre-initialized VectorStore instance
        """
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = VectorStore(
                persist_directory=persist_dir,
                embedding_model=embedding_model,
            )

        groq_api_key = os.getenv("Groq_Api_key") or os.getenv("GROQ_API_KEY") or ""
        if not groq_api_key:
            print("[WARNING] No Groq API key found in environment variables!")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

        self.chat_history: List[Dict[str, str]] = []

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str = None,
    ) -> str:
        """
        Search and summarize — simple interface returning just the answer string.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            doc_filter: Optional source_file name to scope search

        Returns:
            The LLM-generated answer string
        """
        result = self.search_with_trace(query, top_k=top_k, doc_filter=doc_filter)
        return result["answer"]

    def search_with_trace(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str = None,
        chat_history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search with full retrieval trace for the UI.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            doc_filter: Optional source_file name to scope search
            chat_history: Optional conversation history for multi-turn

        Returns:
            Dict with: answer, sources, retrieved_chunks, latency_ms, query
        """
        start_time = time.time()

        # Retrieve relevant chunks
        results = self.vectorstore.query(
            query, top_k=top_k, doc_filter=doc_filter
        )

        if not results:
            return {
                "answer": "No relevant documents found. Please upload some documents first.",
                "sources": [],
                "retrieved_chunks": [],
                "latency_ms": round((time.time() - start_time) * 1000),
                "query": query,
                "chunks_count": 0,
            }

        # Build context from retrieved chunks
        context = "\n\n".join([r["content"] for r in results])

        # Build sources list for citations
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append(
                {
                    "source_file": meta.get(
                        "source_file", meta.get("source", "unknown")
                    ),
                    "page": meta.get("page", "N/A"),
                    "page_label": meta.get("page_label", ""),
                    "score": round(r.get("similarity_score", 0), 4),
                    "content_preview": r["content"][:300] + "..."
                    if len(r["content"]) > 300
                    else r["content"],
                    "full_content": r["content"],
                }
            )

        # Build prompt with optional chat history
        history_context = ""
        if chat_history:
            history_lines = []
            for msg in chat_history[-4:]:  # Last 4 turns for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role.capitalize()}: {content}")
            history_context = (
                "\n\nPrevious conversation:\n" + "\n".join(history_lines) + "\n"
            )

        prompt = f"""You are an intelligent, helpful document assistant. Answer the user's question clearly and concisely based ONLY on the provided context.

Follow these strict formatting guidelines to make the answer visually engaging and easy to read:
- Highlight key metrics, names, and concepts in **bold**.
- Structure details using clear **bullet points** or **numbered lists**.
- Keep paragraphs short and utilize **subheadings** (e.g. `### Summary`, `### Details`) for longer explanations.
- Be polite, direct, and helpful.
- If the context doesn't contain the answer, say so clearly. Do not make up information.

{history_context}
Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.llm.invoke([prompt])
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}"

        latency_ms = round((time.time() - start_time) * 1000)

        # Store in chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results,
            "latency_ms": latency_ms,
            "query": query,
            "chunks_count": len(results),
        }

    def get_document_names(self) -> List[str]:
        """Return list of unique document names in the store"""
        return self.vectorstore.get_document_names()

    def get_collection_count(self) -> int:
        """Return total chunks in the store"""
        return self.vectorstore.get_collection_count()

    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    result = rag_search.search_with_trace(query, top_k=3)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print(f"Latency: {result['latency_ms']}ms")