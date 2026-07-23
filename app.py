"""Streamlit UI: upload a PDF, chat with it, get page-cited answers."""

import streamlit as st

from rag.chat import answer_question
from rag.ingest import build_index, chunk_pages, extract_pages, retrieve

st.set_page_config(page_title="DocuChat — PDF RAG", page_icon="📄")

st.title("📄 DocuChat")
st.caption(
    "Retrieval-Augmented Generation: upload a PDF, ask questions, and get "
    "answers grounded in the document with page citations."
)

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    top_k = st.slider("Chunks retrieved per question", 3, 10, 5)
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n\n"
        "1. Extract text per page (pypdf)\n"
        "2. Chunk with overlap\n"
        "3. Embed → ChromaDB (MiniLM)\n"
        "4. Retrieve top-k per question\n"
        "5. Llama 3.3 answers with [p. N] citations"
    )

if uploaded is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Ingest once per file; keep the index in session state
if st.session_state.get("doc_name") != uploaded.name:
    with st.spinner("Indexing document..."):
        pages = extract_pages(uploaded)
        if not pages:
            st.error("No extractable text found — is this a scanned PDF?")
            st.stop()
        chunks = chunk_pages(pages)
        st.session_state.collection = build_index(chunks, uploaded.name)
        st.session_state.doc_name = uploaded.name
        st.session_state.messages = []
    st.success(f"Indexed {len(pages)} pages into {len(chunks)} chunks.")

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        retrieved = retrieve(st.session_state.collection, question, k=top_k)
        # History excludes the question just asked — answer_question adds it
        # along with the retrieved context.
        history = st.session_state.messages[:-1]
        response = st.write_stream(answer_question(question, retrieved, history))

        with st.expander("Retrieved excerpts"):
            for c in retrieved:
                st.markdown(f"**p. {c['page']}** — {c['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})
