import os
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

# Ensure the src package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_single_file
from src.vectorstore import VectorStore
from src.search import RAGSearch


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Document RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean, High-Contrast CSS (Fully compatible with Light & Dark Themes) ────
st.markdown(
    """
<style>
    /* Max width of content container for readability */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* HCI Visual Banner using theme-compliant values */
    .banner-container {
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .banner-title {
        margin: 0 0 0.5rem 0 !important;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(to right, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .banner-subtitle {
        margin: 0 !important;
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Status Badges with High Contrast Colors */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid transparent;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border-color: rgba(239, 68, 68, 0.3);
    }

    /* Custom styles for expander headers */
    .expander-header {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "vectorstore" not in st.session_state:
        # Unique collection per browser tab session prevents race conditions and isolates user uploads
        st.session_state.vectorstore = VectorStore(
            collection_name=f"user_uploads_{st.session_state.session_id}"
        )
    if "rag_search" not in st.session_state:
        st.session_state.rag_search = RAGSearch(
            vectorstore=st.session_state.vectorstore
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 4
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 500
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100


init_session_state()


# ── Helper: Process uploaded file ────────────────────────────────────────────
def process_uploaded_file(uploaded_file):
    """Save uploaded file to temp dir, load, chunk, embed, and add to store."""
    doc_info = {
        "name": uploaded_file.name,
        "status": "Processing ⏳",
        "pages": 0,
        "chunks": 0,
        "error": None,
    }

    try:
        # Save to temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Load documents from the file
        documents = load_single_file(tmp_path)
        doc_info["pages"] = len(documents)

        if not documents:
            doc_info["status"] = "Failed ❌"
            doc_info["error"] = "No content extracted from file"
            return doc_info

        # Clean metadata: replace temp file name with the original user-facing file name!
        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name

        # Chunk and add to vector store
        chunks = st.session_state.vectorstore.build_from_documents(
            documents,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
        doc_info["chunks"] = len(chunks)
        doc_info["status"] = "Indexed ✅"

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    except Exception as e:
        doc_info["status"] = "Failed ❌"
        doc_info["error"] = str(e)

    return doc_info


# ── Sidebar Layout ───────────────────────────────────────────────────────────
with st.sidebar:
    # Sidebar header with clean branding
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;'>
            <span style='font-size: 2.2rem;'>🤖</span>
            <h2 style='margin: 0; font-size: 1.5rem; font-weight: 800;'>RAG Control Panel</h2>
        </div>
        <p style='font-size: 0.85rem; opacity: 0.7;'>HCI Optimized Workspace</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── File Upload ──
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, CSV, XLSX)",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        existing_names = {d["name"] for d in st.session_state.uploaded_docs}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            with st.spinner(f"Processing {len(new_files)} document(s)..."):
                for uploaded_file in new_files:
                    doc_info = process_uploaded_file(uploaded_file)
                    st.session_state.uploaded_docs.append(doc_info)
            st.rerun()

    st.divider()

    # ── Document Library ──
    st.markdown("### 📋 Document Library")
    total_chunks = st.session_state.vectorstore.get_collection_count()

    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            status_class = "badge-success" if "Indexed" in doc["status"] else ("badge-warning" if "Processing" in doc["status"] else "badge-error")
            status_text = "Indexed" if "Indexed" in doc["status"] else ("Processing" if "Processing" in doc["status"] else "Failed")
            
            # Using Streamlit components directly for theme-compatible card rendering
            with st.container(border=True):
                col_n, col_b = st.columns([2, 1])
                with col_n:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"📄 {doc['pages']} pgs · 🧱 {doc['chunks']} chk")
                with col_b:
                    st.markdown(f"<span class='badge {status_class}'>{status_text}</span>", unsafe_allow_html=True)
            
            if doc.get("error"):
                st.error(doc["error"], icon="⚠️")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # ── Interactive RAG Tuning ──
    with st.expander("⚙️ RAG Settings"):
        st.session_state.top_k = st.slider(
            "Retrieve Chunks (Top K)",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
        )
        st.session_state.chunk_size = st.slider(
            "Chunk Size (chars)",
            min_value=100,
            max_value=2000,
            value=st.session_state.chunk_size,
            step=50,
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=500,
            value=st.session_state.chunk_overlap,
            step=10,
        )

    # ── Search Scope ──
    doc_names = st.session_state.rag_search.get_document_names()
    if doc_names:
        st.markdown("### 🔍 Search Scope")
        scope_options = ["All Documents"] + doc_names
        selected_scope = st.selectbox(
            "Limit search scope:",
            scope_options,
            index=0,
            key="search_scope",
            label_visibility="collapsed",
        )
    else:
        selected_scope = "All Documents"

    st.divider()

    # ── Engine Metrics ──
    st.markdown("### 📊 Engine Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", total_chunks)
    with col2:
        st.metric("Total Turns", len(st.session_state.chat_history) // 2)

    if st.session_state.chat_history:
        last_result = st.session_state.chat_history[-1].get("metadata", {})
        if last_result:
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Last Latency", f"{last_result.get('latency_ms', 0)} ms")
            with col4:
                st.metric("Chunks Read", last_result.get("chunks_count", 0))

    st.divider()

    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.rag_search.clear_history()
        st.success("Memory cleared!")
        st.rerun()


# ── Main Layout ──────────────────────────────────────────────────────────────
# Interactive Light/Dark Banner Header
st.markdown(
    """
    <div class="banner-container">
        <h1 class="banner-title">Document RAG Assistant</h1>
        <p class="banner-subtitle">Upload multi-format documents, select context scopes, ask questions, and explore citations dynamically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Onboarding steps for visual guidance
if total_chunks == 0:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#3b82f6; font-size: 1.15rem; font-weight: 700;'>💡 Step-by-Step Onboarding Guide</h3>", unsafe_allow_html=True)
        st.markdown("1️⃣ &nbsp;**Upload** your files (PDF, DOCX, TXT, CSV, or XLSX) using the sidebar file picker.")
        st.markdown("2️⃣ &nbsp;Wait for the loader to output the `Indexed ✅` badge in your document library.")
        st.markdown("3️⃣ &nbsp;Optionally adjust settings like **Top K retrieval count** or filter scope in the sidebar.")
        st.markdown("4️⃣ &nbsp;Write your query in the input field below.")
        st.markdown("5️⃣ &nbsp;Examine **citations** and **retrieval trace calculations** directly beneath the response.")

# ── Display Conversational Dialogue ──
for i, msg in enumerate(st.session_state.chat_history):
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            sources = metadata.get("sources", [])
            chunks = metadata.get("retrieved_chunks", [])
            latency = metadata.get("latency_ms")

            # Citations Expandable Panel - Clean & Contrast Safe in Light and Dark Mode
            if sources:
                with st.expander(f"📎 Source Citations ({len(sources)})", expanded=False):
                    for j, src in enumerate(sources):
                        # Clean source file name from temp folder paths if any
                        display_source = Path(src['source_file']).name
                        st.markdown(f"**Source [{j+1}]:** `{display_source}` &nbsp;|&nbsp; **Page:** `{src['page']}` &nbsp;|&nbsp; **Relevance:** `{src['score']}`")
                        st.info(src['full_content'])

            # Debug Trace Panel
            if chunks:
                with st.expander(
                    f"🔍 Retrieval Trace ({len(chunks)} chunks)", expanded=False
                ):
                    for j, chunk in enumerate(chunks):
                        meta = chunk.get("metadata", {})
                        score = chunk.get("similarity_score", 0)
                        
                        display_chunk_source = Path(meta.get('source_file', 'unknown')).name
                        st.markdown(f"🧱 **Chunk {j+1}:** `{display_chunk_source}` &nbsp;|&nbsp; **Page:** `{meta.get('page', 'N/A')}` &nbsp;|&nbsp; **Cosine Score:** `{score:.4f}`")
                        st.code(chunk.get("content", "")[:500], language=None)

            if latency:
                st.caption(f"⏱️ Generated in {latency}ms")


# ── Chat Action & Input ──
if prompt := st.chat_input("Ask a question about your documents..."):
    if total_chunks == 0:
        st.warning("⚠️ Index is empty. Please upload some documents first.")
        st.stop()

    # User message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Scope filtering
    doc_filter = None
    if selected_scope != "All Documents":
        doc_filter = selected_scope

    # History
    history_for_context = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history[:-1]
    ]

    # Assistant Response Generation
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and formulating response..."):
            result = st.session_state.rag_search.search_with_trace(
                query=prompt,
                top_k=st.session_state.top_k,
                doc_filter=doc_filter,
                chat_history=history_for_context,
            )

        st.markdown(result["answer"])

        # Display citations in the active response block
        sources = result.get("sources", [])
        if sources:
            with st.expander(
                f"📎 Source Citations ({len(sources)})", expanded=False
            ):
                for j, src in enumerate(sources):
                    display_source = Path(src['source_file']).name
                    st.markdown(f"**Source [{j+1}]:** `{display_source}` &nbsp;|&nbsp; **Page:** `{src['page']}` &nbsp;|&nbsp; **Relevance:** `{src['score']}`")
                    st.info(src['full_content'])

        # Display trace in the active response block
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            with st.expander(
                f"🔍 Retrieval Trace ({len(chunks)} chunks)", expanded=False
            ):
                for j, chunk in enumerate(chunks):
                    meta = chunk.get("metadata", {})
                    score = chunk.get("similarity_score", 0)
                    display_chunk_source = Path(meta.get('source_file', 'unknown')).name
                    st.markdown(f"🧱 **Chunk {j+1}:** `{display_chunk_source}` &nbsp;|&nbsp; **Page:** `{meta.get('page', 'N/A')}` &nbsp;|&nbsp; **Cosine Score:** `{score:.4f}`")
                    st.code(chunk.get("content", "")[:500], language=None)

        st.caption(f"⏱️ Generated in {result['latency_ms']}ms")

    # Add message to history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "latency_ms": result["latency_ms"],
                "chunks_count": result["chunks_count"],
            },
        }
    )
