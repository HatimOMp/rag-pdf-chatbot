import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG PDF Chatbot")
st.markdown(
    "Upload any PDF and ask questions about it. "
    "Powered by **Mistral AI** embeddings + **FAISS** vector search + **RAG** pipeline."
)

# ── Session state ────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    top_k = st.slider(
        "Chunks to retrieve per query", 2, 8, 4,
        help="More chunks = more context but slower response"
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # Process PDF
                processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
                chunks, num_pages = processor.process(tmp_path)

                # Build vector store
                vector_store = VectorStore()
                num_chunks = vector_store.build_index(chunks)

                # Initialize RAG engine
                rag_engine = RAGEngine(vector_store)

                # Save to session
                st.session_state.vector_store = vector_store
                st.session_state.rag_engine = rag_engine
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.num_chunks = num_chunks
                st.session_state.num_pages = num_pages

                st.success(f"✅ Ready!")

            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown(f"**File:** {st.session_state.pdf_name}")
        st.metric("Pages", st.session_state.num_pages)
        st.metric("Chunks", st.session_state.num_chunks)

        if st.button("🗑️ Reset conversation"):
            st.session_state.chat_history = []
            if st.session_state.rag_engine:
                st.session_state.rag_engine.reset_conversation()
            st.rerun()

# ── Main chat interface ───────────────────────────────────────────────
if not st.session_state.pdf_name:
    st.info("👈 Upload a PDF in the sidebar to get started.")

    st.markdown("## 💡 What can this chatbot do?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📥 Upload any PDF**
        Research papers, legal contracts,
        financial reports, manuals,
        books — anything.
        """)
    with col2:
        st.markdown("""
        **🔍 Semantic Search**
        Uses Mistral AI embeddings
        and FAISS vector search to find
        the most relevant passages.
        """)
    with col3:
        st.markdown("""
        **💬 Cited Answers**
        Every answer includes the
        page numbers where the
        information was found.
        """)

else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    st.caption(f"📄 Sources: pages {message['sources']}")

    # Chat input
    query = st.chat_input("Ask a question about your document...")

    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = st.session_state.rag_engine.answer(
                        query, top_k=top_k
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                    chunks = result["retrieved_chunks"]

                    st.markdown(answer)

                    if sources:
                        st.caption(f"📄 Sources: pages {sources}")

                    # Show retrieved chunks in expander
                    with st.expander("🔍 Retrieved context chunks"):
                        for i, chunk in enumerate(chunks):
                            st.markdown(
                                f"**Chunk {i+1}** "
                                f"(Page {chunk['chunk']['page']}, "
                                f"Score: {chunk['score']:.3f})"
                            )
                            st.text(chunk['chunk']['text'][:300] + "...")
                            st.divider()

                except Exception as e:
                    answer = f"Error generating answer: {e}"
                    sources = []
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })